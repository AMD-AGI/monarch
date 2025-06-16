/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::future::Future;
use std::future::pending;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorHandle;
use hyperactor::ActorId;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use hyperactor::PortId;
use hyperactor::forward;
use hyperactor::message::Bind;
use hyperactor::message::Bindings;
use hyperactor::message::IndexedErasedUnbound;
use hyperactor::message::Unbind;
use hyperactor_mesh::actor_mesh::Cast;
use monarch_types::PickledPyObject;
use monarch_types::SerializablePyErr;
use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::PyBaseException;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::types::PyType;
use serde::Deserialize;
use serde::Serialize;
use serde_bytes::ByteBuf;
use tokio::runtime::Handle;
use tokio::sync::Mutex;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::mailbox::PyMailbox;
use crate::proc::InstanceWrapper;
use crate::proc::PyActorId;
use crate::proc::PyProc;
use crate::proc::PySerialized;
use crate::runtime::signal_safe_block_on;
use crate::shape::PyShape;

#[pyclass(frozen, module = "monarch._rust_bindings.monarch_hyperactor.actor")]
#[derive(Serialize, Deserialize, Named)]
pub struct PickledMessage {
    sender_actor_id: ActorId,
    message: ByteBuf,
}

impl std::fmt::Debug for PickledMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PickledMessage(sender_actor_id: {:?} message: {})",
            self.sender_actor_id,
            hyperactor::data::HexFmt(self.message.as_slice()),
        )
    }
}

#[pymethods]
impl PickledMessage {
    #[new]
    #[pyo3(signature = (*, sender_actor_id, message))]
    fn new(sender_actor_id: &PyActorId, message: Vec<u8>) -> Self {
        Self {
            sender_actor_id: sender_actor_id.into(),
            message: ByteBuf::from(message),
        }
    }

    #[getter]
    fn sender_actor_id(&self) -> PyActorId {
        self.sender_actor_id.clone().into()
    }

    #[getter]
    fn message<'a>(&self, py: Python<'a>) -> Bound<'a, PyBytes> {
        PyBytes::new(py, self.message.as_ref())
    }

    fn serialize(&self) -> PyResult<PySerialized> {
        PySerialized::new(self)
    }
}

#[pyclass(module = "monarch._rust_bindings.monarch_hyperactor.actor")]
pub struct PickledMessageClientActor {
    instance: Arc<Mutex<InstanceWrapper<PickledMessage>>>,
}

#[pymethods]
impl PickledMessageClientActor {
    #[new]
    fn new(proc: &PyProc, actor_name: &str) -> PyResult<Self> {
        Ok(Self {
            instance: Arc::new(Mutex::new(InstanceWrapper::new(proc, actor_name)?)),
        })
    }

    /// Send a message to any actor that can receive the corresponding serialized message.
    fn send(&self, actor_id: &PyActorId, message: &PySerialized) -> PyResult<()> {
        let instance = self.instance.blocking_lock();
        instance.send(actor_id, message)
    }

    /// Get the next message from the queue. It will block until a message is received
    /// or the timeout is reached in which case it will return None
    /// If the actor has been stopped, this returns an error.
    #[pyo3(signature = (*, timeout_msec = None))]
    fn get_next_message<'py>(
        &mut self,
        py: Python<'py>,
        timeout_msec: Option<u64>,
    ) -> PyResult<PyObject> {
        let instance = self.instance.clone();
        let result = signal_safe_block_on(py, async move {
            instance.lock().await.next_message(timeout_msec).await
        })?;
        Python::with_gil(|py| {
            result
                .map(|res| res.into_py_any(py))?
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))
        })
    }

    /// Stop the background task and return any messages that were received.
    /// TODO: This is currently just aborting the task, we should have a better way to stop it.
    fn drain_and_stop<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut instance = self.instance.blocking_lock();
        let messages = instance
            .drain_and_stop()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .into_iter()
            .map(|message| message.into_py_any(py))
            .collect::<PyResult<Vec<PyObject>>>()?;
        PyList::new(py, messages)
    }

    fn world_status<'py>(&mut self, py: Python<'py>) -> PyResult<PyObject> {
        let instance = self.instance.clone();

        let worlds = signal_safe_block_on(py, async move {
            instance.lock().await.world_status(Default::default()).await
        })??;
        Python::with_gil(|py| {
            let py_dict = PyDict::new(py);
            for (world, status) in worlds {
                py_dict.set_item(world.to_string(), status.to_string())?;
            }
            Ok(py_dict.into())
        })
    }

    #[getter]
    fn actor_id(&self) -> PyResult<PyActorId> {
        let instance = self.instance.blocking_lock();
        Ok(PyActorId::from(instance.actor_id().clone()))
    }
}

#[pyclass(frozen, module = "monarch._rust_bindings.monarch_hyperactor.actor")]
#[derive(Clone, Serialize, Deserialize, Named, PartialEq)]
pub struct PythonMessage {
    method: String,
    message: ByteBuf,
    response_port: Option<PortId>,
    rank: Option<usize>,
}

impl std::fmt::Debug for PythonMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PythonMessage")
            .field("method", &self.method)
            .field(
                "message",
                &hyperactor::data::HexFmt(self.message.as_slice()).to_string(),
            )
            .finish()
    }
}

impl Unbind for PythonMessage {
    fn bindings(&self) -> anyhow::Result<Bindings> {
        let mut bindings = Bindings::default();
        if let Some(response_port) = &self.response_port {
            bindings.push(response_port)?;
        }
        Ok(bindings)
    }
}

impl Bind for PythonMessage {
    fn bind(mut self, bindings: &Bindings) -> anyhow::Result<Self> {
        if let Some(response_port) = &mut self.response_port {
            bindings.rebind::<PortId>([response_port].into_iter())?;
        }
        Ok(self)
    }
}

#[pymethods]
impl PythonMessage {
    #[new]
    #[pyo3(signature = (method, message, response_port, rank))]
    fn new(
        method: String,
        message: Vec<u8>,
        response_port: Option<crate::mailbox::PyPortId>,
        rank: Option<usize>,
    ) -> Self {
        Self {
            method,
            message: ByteBuf::from(message),
            response_port: response_port.map(Into::into),
            rank,
        }
    }

    #[getter]
    fn method(&self) -> &String {
        &self.method
    }

    #[getter]
    fn message<'a>(&self, py: Python<'a>) -> Bound<'a, PyBytes> {
        PyBytes::new(py, self.message.as_ref())
    }

    #[getter]
    fn response_port(&self) -> Option<crate::mailbox::PyPortId> {
        self.response_port.clone().map(Into::into)
    }

    #[getter]
    fn rank(&self) -> Option<usize> {
        self.rank
    }
}

#[pyclass(module = "monarch._rust_bindings.monarch_hyperactor.actor")]
pub(super) struct PythonActorHandle {
    pub(super) inner: ActorHandle<PythonActor>,
}

#[pymethods]
impl PythonActorHandle {
    // TODO: do the pickling in rust
    fn send(&self, message: &PythonMessage) -> PyResult<()> {
        self.inner
            .send(message.clone())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok(())
    }

    fn bind(&self) -> PyActorId {
        self.inner.bind::<PythonActor>().into_actor_id().into()
    }
}

/// Controls how actor method dispatch happens.
///
/// In order to preserve the expected behavior of things like thread-local
/// storage, we need to perform dispatch on a consistent thread.
#[derive(Debug)]
enum DispatchType {
    /// This actor is synchronous (i.e. only has `def` endpoints). Dispatch via
    /// running Python code inline on the dedicated actor thread.
    Sync,
    /// This actor is asynchronous (i.e. has at least one `async def` endpoint).
    /// Dispatch by running on a dedicated thread with an asyncio event loop.
    Async(pyo3_async_runtimes::TaskLocals),
}

/// An actor for which message handlers are implemented in Python.
#[derive(Debug)]
#[hyperactor::export_spawn(PythonMessage, Cast<PythonMessage>, IndexedErasedUnbound<Cast<PythonMessage>>)]
pub(super) struct PythonActor {
    /// The Python object that we delegate message handling to. An instance of
    /// `monarch.actor_mesh._Actor` or `_AsyncActor`.
    pub(super) actor: PyObject,

    /// How to dispatch methods on this actor
    dispatch_type: DispatchType,
}

#[async_trait]
impl Actor for PythonActor {
    type Params = PickledPyObject;

    async fn new(actor_type: PickledPyObject) -> Result<Self, anyhow::Error> {
        Ok(Python::with_gil(|py| -> Result<Self, SerializablePyErr> {
            let unpickled = actor_type.unpickle(py)?;
            let class_type: &Bound<'_, PyType> = unpickled.downcast()?;
            let actor = class_type.call0()?.into_pyobject(py)?;

            let actor_mesh_module = Python::import(py, "monarch.actor_mesh")?;
            let actor_class = actor_mesh_module.getattr("_AsyncActor")?;
            let is_async = actor.is_instance(&actor_class)?;

            let dispatch_type = if is_async {
                // Get the event loop state to run PythonActor handlers in. We construct a
                // fresh event loop in its own thread for us to schedule this work onto, to
                // avoid disturbing any event loops that the user might be running.
                //
                // First, release the GIL so that the thread spawned below can acquire it.
                DispatchType::Async(Python::allow_threads(py, || {
                    let (tx, rx) = std::sync::mpsc::channel();
                    let _ = std::thread::spawn(move || {
                        Python::with_gil(|py| {
                            let asyncio = Python::import(py, "asyncio").unwrap();
                            let event_loop = asyncio.call_method0("new_event_loop").unwrap();
                            asyncio
                                .call_method1("set_event_loop", (event_loop.clone(),))
                                .unwrap();

                            let task_locals =
                                pyo3_async_runtimes::TaskLocals::new(event_loop.clone())
                                    .copy_context(py)
                                    .unwrap();
                            tx.send(task_locals).unwrap();
                            event_loop.call_method0("run_forever").unwrap();
                        });
                    });
                    rx.recv().unwrap()
                }))
            } else {
                DispatchType::Sync
            };

            Ok(Self {
                actor: actor.into(),
                dispatch_type,
            })
        })?)
    }

    /// Specialize spawn_server_task for PythonActor, because we want to run the stream on a
    /// dedicated OS thread. We do this to guarantee tha all Python code is
    /// executed on the same thread, since often Python code uses thread-local
    /// state or otherwise assumes that it is called only from a single thread.
    fn spawn_server_task<F>(future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let (join_tx, join_rx) = tokio::sync::oneshot::channel();
        // It is important that we spawn a standalone thread for the work here,
        // as opposed to using `spawn_blocking` to spawn a tokio-managed thread.
        // This is because the worker stream may call uninterruptible FFI code
        // that can deadlock (CUDA, NCCL).
        // If we use a tokio-managed blocking thread, then runtime teardown will
        // try to wait for tasks on that thread to reach an await point, and
        // hang forever.
        let builder = std::thread::Builder::new().name("python-actor".to_string());
        let _thread_handle = builder.spawn(move || {
            // Spawn a new thread with a single-threaded tokio runtime to run the
            // actor loop.  We avoid the current-threaded runtime, so that we can
            // use `block_in_place` for nested async-to-sync-to-async flows.
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_io()
                .build()
                .unwrap();
            rt.block_on(async {
                tokio::task::block_in_place(|| {
                    // Allow e.g. destructing py objects on this thread, which
                    // can happen at shutdown when the a stream actors env map
                    // for rvalues is dropped (e.g. P1673311499).
                    // https://github.com/PyO3/pyo3/discussions/3499
                    Python::with_gil(|py| {
                        py.allow_threads(|| {
                            let result = Handle::current().block_on(future);
                            if join_tx.send(result).is_err() {
                                panic!("could not send join result")
                            }
                        })
                    })
                })
            })
        });

        // In order to bridge the synchronous join handle with the async world,
        // smuggle the result through a channel.
        tokio::spawn(async move { join_rx.await.unwrap() })
    }
}

// [Panics in async endpoints]
// This class exists to solve a deadlock when an async endpoint calls into some
// Rust code that panics.
//
// When an async endpoint is invoked and calls into Rust, the following sequence happens:
//
// hyperactor message -> PythonActor::handle() -> call _Actor.handle() in Python
//   -> convert the resulting coroutine into a Rust future, but scheduled on
//      the Python asyncio event loop (`into_future_with_locals`)
//   -> set a callback on Python asyncio loop to ping a channel that fulfills
//      the Rust future when the Python coroutine has finished. ('PyTaskCompleter`)
//
// This works fine for normal results and Python exceptions: we will take the
// result of the callback and send it through the channel, where it will be
// returned to the `await`er of the Rust future.
//
// This DOESN'T work for panics. The behavior of a panic in pyo3-bound code is
// that it will get caught by pyo3 and re-thrown to Python as a PanicException.
// And if that PanicException ever makes it back to Rust, it will get unwound
// instead of passed around as a normal PyErr type.
//
// So:
//   - Endpoint panics.
//   - This panic is captured as a PanicException in Python and
//     stored as the result of the Python asyncio task.
//   - When the callback in `PyTaskCompleter` queries the status of the task to
//     pass it back to the Rust awaiter, instead of getting a Result type, it
//     just starts resumes unwinding the PanicException
//   - This triggers a deadlock, because the whole task dies without ever
//     pinging the response channel, and the Rust awaiter will never complete.
//
// We work around this by passing a side-channel to our Python task so that it,
// in Python, can catch the PanicException and notify the Rust awaiter manually.
// In this way we can guarantee that the awaiter will complete even if the
// `PyTaskCompleter` callback explodes.
#[pyclass(module = "monarch._rust_bindings.monarch_hyperactor.actor")]
struct PanicFlag {
    sender: Option<tokio::sync::oneshot::Sender<PyObject>>,
}

#[pymethods]
impl PanicFlag {
    fn signal_panic(&mut self, ex: PyObject) {
        self.sender.take().unwrap().send(ex).unwrap();
    }
}

#[async_trait]
impl Handler<PythonMessage> for PythonActor {
    async fn handle(
        &mut self,
        this: &Instance<Self>,
        message: PythonMessage,
    ) -> anyhow::Result<()> {
        let mailbox = PyMailbox {
            inner: this.mailbox_for_py().clone(),
        };
        match &self.dispatch_type {
            DispatchType::Sync => {
                Python::with_gil(|py| -> Result<_, SerializablePyErr> {
                    tokio::task::block_in_place(|| {
                        self.actor
                            .call_method(py, "handle", (mailbox, message), None)
                            .map_err(|err| err.into())
                    })
                })?;
            }
            DispatchType::Async(task_locals) => {
                // Create a channel for signaling panics in async endpoints.
                // See [Panics in async endpoints].
                let (sender, receiver) = oneshot::channel();

                let future = Python::with_gil(|py| -> Result<_, SerializablePyErr> {
                    let awaitable = self.actor.call_method(
                        py,
                        "handle",
                        (
                            mailbox,
                            message,
                            PanicFlag {
                                sender: Some(sender),
                            },
                        ),
                        None,
                    )?;
                    pyo3_async_runtimes::into_future_with_locals(
                        task_locals,
                        awaitable.into_bound(py),
                    )
                    .map_err(|err| err.into())
                })?;
                let handler = AsyncEndpointTask::spawn(this, ()).await?;
                handler.run(this, PythonTask::new(future), receiver).await?;
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Handler<Cast<PythonMessage>> for PythonActor {
    async fn handle(
        &mut self,
        this: &Instance<Self>,
        Cast {
            message,
            rank,
            shape,
        }: Cast<PythonMessage>,
    ) -> anyhow::Result<()> {
        let mailbox = PyMailbox {
            inner: this.mailbox_for_py().clone(),
        };
        match &self.dispatch_type {
            DispatchType::Sync => {
                Python::with_gil(|py| -> Result<_, SerializablePyErr> {
                    tokio::task::block_in_place(|| {
                        self.actor
                            .call_method(
                                py,
                                "handle_cast",
                                (mailbox, rank.0, PyShape::from(shape), message),
                                None,
                            )
                            .map_err(|err| err.into())
                    })
                })?;
            }
            DispatchType::Async(task_locals) => {
                // Create a channel for signaling panics in async endpoints.
                // See [Panics in async endpoints].
                let (sender, receiver) = oneshot::channel();

                let future = Python::with_gil(|py| -> Result<_, SerializablePyErr> {
                    let awaitable = self.actor.call_method(
                        py,
                        "handle_cast",
                        (
                            mailbox,
                            rank.0,
                            PyShape::from(shape),
                            message,
                            PanicFlag {
                                sender: Some(sender),
                            },
                        ),
                        None,
                    )?;

                    pyo3_async_runtimes::into_future_with_locals(
                        task_locals,
                        awaitable.into_bound(py),
                    )
                    .map_err(|err| err.into())
                })?;

                let handler = AsyncEndpointTask::spawn(this, ()).await?;
                handler.run(this, PythonTask::new(future), receiver).await?;
            }
        };
        Ok(())
    }
}

/// Helper struct to make a Python future passable in an actor message.
///
/// Also so that we don't have to write this massive type signature everywhere
struct PythonTask {
    future: Arc<Mutex<Option<Pin<Box<dyn Future<Output = PyResult<PyObject>> + Send + 'static>>>>>,
}

impl PythonTask {
    fn new(fut: impl Future<Output = PyResult<PyObject>> + Send + 'static) -> Self {
        Self {
            future: Arc::new(Mutex::new(Some(Box::pin(fut)))),
        }
    }

    async fn take(self) -> Pin<Box<dyn Future<Output = PyResult<PyObject>> + Send + 'static>> {
        self.future.lock().await.take().unwrap()
    }
}

impl fmt::Debug for PythonTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PythonTask")
            .field("future", &"<PythonFuture>")
            .finish()
    }
}

/// An ['Actor'] used to monitor the result of an async endpoint. We use an
/// actor so that:
/// - Actually waiting on the async endpoint can happen concurrently with other endpoints.
/// - Any uncaught errors in the async endpoint will get propagated as a supervision event.
#[derive(Debug)]
struct AsyncEndpointTask {}

/// An invocation of an async endpoint on a [`PythonActor`].
#[derive(Handler, HandleClient, Debug)]
enum AsyncEndpointInvocation {
    Run(PythonTask, oneshot::Receiver<PyObject>),
}

#[async_trait]
impl Actor for AsyncEndpointTask {
    type Params = ();

    async fn new(_params: Self::Params) -> anyhow::Result<Self> {
        Ok(Self {})
    }
}

#[async_trait]
#[forward(AsyncEndpointInvocation)]
impl AsyncEndpointInvocationHandler for AsyncEndpointTask {
    async fn run(
        &mut self,
        this: &Instance<Self>,
        task: PythonTask,
        side_channel: oneshot::Receiver<PyObject>,
    ) -> anyhow::Result<()> {
        // Drive our PythonTask to completion, but listen on the side channel
        // and raise an error if we hear anything there.

        let err_or_never = async {
            // The side channel will resolve with a value if a panic occured during
            // processing of the async endpoint, see [Panics in async endpoints].
            match side_channel.await {
                Ok(value) => Python::with_gil(|py| -> Result<(), SerializablePyErr> {
                    let err: PyErr = value
                        .downcast_bound::<PyBaseException>(py)
                        .unwrap()
                        .clone()
                        .into();
                    Err(SerializablePyErr::from(py, &err))
                }),
                // An Err means that the sender has been dropped without sending.
                // That's okay, it just means that the Python task has completed.
                // In that case, just never resolve this future. We expect the other
                // branch of the select to finish eventually.
                Err(_) => pending().await,
            }
        };
        let future = task.take().await;
        let result: Result<(), SerializablePyErr> = tokio::select! {
            result = future => {
                match result {
                    Ok(_) => Ok(()),
                    Err(e) => Err(e.into()),
                }
            },
            result = err_or_never => {
                result
            }
        };
        result?;

        // Stop this actor now that its job is done.
        this.stop()?;
        Ok(())
    }
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PickledMessage>()?;
    hyperactor_mod.add_class::<PickledMessageClientActor>()?;
    hyperactor_mod.add_class::<PythonActorHandle>()?;
    hyperactor_mod.add_class::<PythonMessage>()?;
    hyperactor_mod.add_class::<PanicFlag>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use hyperactor::id;

    use super::*;

    #[test]
    fn test_python_message_bind_unbind() {
        let message = PythonMessage {
            method: "test".to_string(),
            message: ByteBuf::from(vec![1, 2, 3]),
            response_port: Some(id!(world[0].client[0][123])),
            rank: None,
        };
        {
            let unbound = message.clone().unbind().unwrap();
            assert_eq!(message, unbound.bind().unwrap());
        }

        let no_port_message = PythonMessage {
            response_port: None,
            ..message
        };
        {
            let unbound = no_port_message.clone().unbind().unwrap();
            assert_eq!(no_port_message, unbound.bind().unwrap());
        }
    }
}
