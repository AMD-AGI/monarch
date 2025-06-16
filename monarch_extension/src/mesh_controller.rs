/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::iter::repeat_n;
use std::sync::Arc;
use std::sync::atomic;
use std::sync::atomic::AtomicUsize;

use hyperactor::ActorRef;
use hyperactor::data::Serialized;
use hyperactor_mesh::actor_mesh::ActorMesh;
use hyperactor_mesh::actor_mesh::RootActorMesh;
use hyperactor_mesh::proc_mesh::SharedSpawnable;
use monarch_hyperactor::ndslice::PySlice;
use monarch_hyperactor::proc::InstanceWrapper;
use monarch_hyperactor::proc::PyActorId;
use monarch_hyperactor::proc::PyProc;
use monarch_hyperactor::proc_mesh::PyProcMesh;
use monarch_hyperactor::runtime::signal_safe_block_on;
use monarch_messages::client::Exception;
use monarch_messages::controller::ControllerActor;
use monarch_messages::controller::ControllerMessage;
use monarch_messages::controller::Seq;
use monarch_messages::debugger::DebuggerAction;
use monarch_messages::debugger::DebuggerActor;
use monarch_messages::debugger::DebuggerMessage;
use monarch_messages::worker::Ref;
use monarch_messages::worker::WorkerMessage;
use monarch_messages::worker::WorkerParams;
use monarch_tensor_worker::AssignRankMessage;
use monarch_tensor_worker::WorkerActor;
use ndslice::Slice;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tokio::sync::Mutex;

use crate::convert::convert;

#[pyclass(
    subclass,
    module = "monarch._rust_bindings.monarch_extension.mesh_controller"
)]
struct _Controller {
    controller_instance: Arc<Mutex<InstanceWrapper<ControllerMessage>>>,
    workers: RootActorMesh<'static, WorkerActor>,
    pending_messages: VecDeque<PyObject>,
    history: History,
}

impl _Controller {
    fn add_responses(
        &mut self,
        py: Python<'_>,
        responses: Vec<(
            monarch_messages::controller::Seq,
            Option<Result<hyperactor::data::Serialized, monarch_messages::client::Exception>>,
        )>,
    ) -> PyResult<()> {
        for (seq, response) in responses {
            let message = crate::client::WorkerResponse::new(seq, response);
            self.pending_messages.push_back(message.into_py(py));
        }
        Ok(())
    }
    fn fill_messages<'py>(&mut self, py: Python<'py>, timeout_msec: Option<u64>) -> PyResult<()> {
        let instance = self.controller_instance.clone();
        let result = signal_safe_block_on(py, async move {
            instance.lock().await.next_message(timeout_msec).await
        })??;
        result.map(|m| self.add_message(m)).transpose()?;
        Ok(())
    }

    fn add_message(&mut self, message: ControllerMessage) -> PyResult<()> {
        Python::with_gil(|py| -> PyResult<()> {
            match message {
                ControllerMessage::DebuggerMessage {
                    debugger_actor_id,
                    action,
                } => {
                    let dm = crate::client::DebuggerMessage::new(debugger_actor_id.into(), action)?
                        .into_py(py);
                    self.pending_messages.push_back(dm);
                }
                ControllerMessage::Status {
                    seq,
                    worker_actor_id,
                    controller: false,
                } => {
                    let rank = worker_actor_id.rank();
                    let responses = self.history.rank_completed(rank, seq);
                    self.add_responses(py, responses)?;
                }
                ControllerMessage::RemoteFunctionFailed { seq, error } => {
                    let responses = self
                        .history
                        .propagate_exception(seq, Exception::Error(seq, seq, error));
                    self.add_responses(py, responses)?;
                }
                ControllerMessage::FetchResult {
                    seq,
                    value: Ok(value),
                } => {
                    self.history.set_result(seq, value);
                }
                ControllerMessage::FetchResult {
                    seq,
                    value: Err(error),
                } => {
                    let responses = self
                        .history
                        .propagate_exception(seq, Exception::Error(seq, seq, error));
                    self.add_responses(py, responses)?;
                }
                message => {
                    panic!("unexpected message: {:?}", message);
                }
            };
            Ok(())
        })
    }
    fn send_slice(&mut self, slice: Slice, message: WorkerMessage) -> PyResult<()> {
        self.workers
            .cast_slices(vec![slice], message)
            .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))
        // let shape = Shape::new(
        //     (0..slice.sizes().len()).map(|i| format!("d{i}")).collect(),
        //     slice,
        // )
        // .unwrap();
        // println!("SENDING TO {:?} {:?}", &shape, &message);
        // let worker_slice = SlicedActorMesh::new(&self.workers, shape);
        // worker_slice
        //     .cast(ndslice::Selection::True, message)
        //     .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))
    }
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

#[pymethods]
impl _Controller {
    #[new]
    fn new(py: Python, py_proc_mesh: &PyProcMesh) -> PyResult<Self> {
        let proc_mesh = py_proc_mesh.inner.as_ref();
        let id = NEXT_ID.fetch_add(1, atomic::Ordering::Relaxed);
        let controller_instance: InstanceWrapper<ControllerMessage> = InstanceWrapper::new(
            &PyProc::new_from_proc(proc_mesh.client_proc().clone()),
            &format!("tensor_engine_controller_{}", id),
        )?;

        let controller_actor_ref =
            ActorRef::<ControllerActor>::attest(controller_instance.actor_id().clone());

        let slice = proc_mesh.shape().slice();
        if !slice.is_contiguous() || slice.offset() != 0 {
            return Err(PyValueError::new_err(
                "NYI: proc mesh for workers must be contiguous and start at offset 0",
            ));
        }
        let world_size = slice.len();
        let param = WorkerParams {
            world_size,
            // Rank assignment is consistent with proc indices.
            rank: 0,
            device_index: Some(0),
            controller_actor: controller_actor_ref,
        };

        let py_proc_mesh = Arc::clone(&py_proc_mesh.inner);
        let workers: anyhow::Result<RootActorMesh<'_, WorkerActor>> =
            signal_safe_block_on(py, async move {
                let workers = py_proc_mesh
                    .spawn(&format!("tensor_engine_workers_{}", id), &param)
                    .await?;
                //workers.cast(ndslice::Selection::True, )?;
                workers.cast_slices(
                    vec![py_proc_mesh.shape().slice().clone()],
                    AssignRankMessage::AssignRank(),
                )?;
                Ok(workers)
            })?;
        Ok(Self {
            workers: workers?,
            controller_instance: Arc::new(Mutex::new(controller_instance)),
            pending_messages: VecDeque::new(),
            history: History::new(world_size),
        })
    }

    fn node<'py>(
        &mut self,
        seq: u64,
        defs: Bound<'py, PyAny>,
        uses: Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let failures = self.history.add_invocation(
            seq.into(),
            uses.iter()?
                .map(|x| Ref::from_py_object(&x?))
                .collect::<PyResult<Vec<Ref>>>()?,
            defs.iter()?
                .map(|x| Ref::from_py_object(&x?))
                .collect::<PyResult<Vec<Ref>>>()?,
        );
        self.add_responses(defs.py(), failures)?;
        Ok(())
    }

    fn drop_refs(&mut self, refs: Vec<Ref>) -> Result<(), anyhow::Error> {
        self.history.delete_invocations_for_refs(refs);
        Ok(())
    }

    fn send<'py>(&mut self, ranks: Bound<'py, PyAny>, message: Bound<'py, PyAny>) -> PyResult<()> {
        let message: WorkerMessage = convert(message)?;
        if let Ok(slice) = ranks.extract::<PySlice>() {
            self.send_slice(slice.into(), message)?;
        } else {
            let slices = ranks.extract::<Vec<PySlice>>()?;
            for (slice, message) in slices.iter().zip(repeat_n(message, slices.len())) {
                self.send_slice(slice.into(), message)?;
            }
        };
        Ok(())
    }

    #[pyo3(signature = (*, timeout_msec = None))]
    fn _get_next_message<'py>(
        &mut self,
        py: Python<'py>,
        timeout_msec: Option<u64>,
    ) -> PyResult<Option<PyObject>> {
        if self.pending_messages.is_empty() {
            self.fill_messages(py, timeout_msec)?;
        }
        Ok(self.pending_messages.pop_front())
    }

    fn _debugger_attach(&mut self, pdb_actor: PyActorId) -> PyResult<()> {
        let pdb_actor: ActorRef<DebuggerActor> = ActorRef::attest(pdb_actor.into());
        pdb_actor
            .send(
                self.controller_instance.blocking_lock().mailbox(),
                DebuggerMessage::Action {
                    action: DebuggerAction::Attach(),
                },
            )
            .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;
        Ok(())
    }

    fn _debugger_write(&mut self, pdb_actor: PyActorId, bytes: Vec<u8>) -> PyResult<()> {
        let pdb_actor: ActorRef<DebuggerActor> = ActorRef::attest(pdb_actor.into());
        pdb_actor
            .send(
                self.controller_instance.blocking_lock().mailbox(),
                DebuggerMessage::Action {
                    action: DebuggerAction::Write { bytes },
                },
            )
            .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;
        Ok(())
    }
    fn _drain_and_stop(&mut self, py: Python<'_>) -> PyResult<()> {
        self.send_slice(
            self.workers.proc_mesh().shape().slice().clone(),
            WorkerMessage::Exit { error: None },
        )?;
        let instance = self.controller_instance.clone();
        let _ = signal_safe_block_on(py, async move { instance.lock().await.drain_and_stop() })??;
        Ok(())
    }
}

pub(crate) fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<_Controller>()?;
    Ok(())
}

/// An invocation tracks a discrete node in the graph of operations executed by
/// the worker based on instructions from the client.
/// It is useful for tracking the dependencies of an operation and propagating
/// failures. In the future this will be used with more data dependency tracking
/// to support better failure handling.
// Allowing dead code until we do something smarter with defs, uses etc.
#[derive(Debug)]
struct Invocation {
    /// The sequence number of the invocation. This should be unique and increasing across all
    /// invocations.
    seq: Seq,
    /// The references that this invocation defines or redefines. Effectively the
    /// output of the invocation.
    defs: Vec<Ref>,
    /// The result of the invocation. This is set when the invocation is completed or
    /// when a failure is inferred. A successful result will always supersede any failure.
    result: Option<Result<Serialized, Exception>>,
    /// The seqs for the invocations that depend on this invocation. Useful for propagating failures.
    users: HashSet<Seq>,
}

impl Invocation {
    fn new(seq: Seq, defs: Vec<Ref>) -> Self {
        Self {
            seq,
            defs,
            result: None,
            users: HashSet::new(),
        }
    }

    fn add_user(&mut self, user: Seq) {
        self.users.insert(user);
    }

    /// Invocation results can only go from valid to failed, or be
    /// set if the invocation result is empty.
    fn set_result(&mut self, result: Serialized) {
        if self.result.is_none() {
            self.result = Some(Ok(result));
        }
    }

    fn set_exception(&mut self, exception: Exception) {
        match exception {
            Exception::Error(_, caused_by, error) => {
                let e = Err(Exception::Error(self.seq, caused_by, error));
                match self.result {
                    Some(Ok(_)) => {
                        self.result = Some(e);
                    }
                    None => {
                        self.result = Some(e);
                    }
                    Some(Err(_)) => {}
                }
            }
            Exception::Failure(_) => {
                tracing::error!(
                    "system failures {:?} can never be assigned for an invocation",
                    exception
                );
            }
        }
    }

    fn exception(&self) -> Option<&Exception> {
        self.result
            .as_ref()
            .map(Result::as_ref)
            .and_then(Result::err)
    }

    #[allow(dead_code)]
    fn value(&self) -> Option<&Serialized> {
        self.result
            .as_ref()
            .map(Result::as_ref)
            .and_then(Result::ok)
    }
}

#[derive(Debug, PartialEq)]
enum RefStatus {
    // The invocation for this ref is still in progress.
    Invoked(Seq),
    // The invocation for this ref has errored.
    Errored(Exception),
}

/// The history of invocations sent by the client to be executed on the workers.
/// This is used to track dependencies between invocations and to propagate exceptions.
/// It purges history for completed invocations to avoid memory bloat.
/// TODO: Revisit this setup around purging refs automatically once we start doing
/// more complex data dependency tracking. We will want to be more aware of things like
/// borrows, drops etc. directly.
#[derive(Debug)]
struct History {
    /// The first incomplete Seq for each rank. This is used to determine which
    /// Seqs are no longer relevant and can be purged from the history.
    first_incomplete_seqs: MinVector<Seq>,
    /// The minimum incomplete Seq across all ranks.
    min_incomplete_seq: Seq,
    /// A map of seq to the invocation that it represents.
    invocations: HashMap<Seq, Invocation>,
    /// A map of reference to the seq for the invocation that defines it. This is used to
    /// compute dependencies between invocations.
    invocation_for_ref: HashMap<Ref, RefStatus>,
    // Refs to be deleted in mark_worker_complete_and_propagate_failures
    marked_for_deletion: HashSet<Ref>,
    // no new sequence numbers should be below this bound. use for
    // sanity checking.
    seq_lower_bound: Seq,
}

/// A vector that keeps track of the minimum value.
#[derive(Debug)]
struct MinVector<T> {
    data: Vec<T>,
    value_counts: BTreeMap<T, usize>,
}

impl<T> MinVector<T>
where
    T: Ord + Copy,
{
    fn new(data: Vec<T>) -> Self {
        let mut value_counts = BTreeMap::new();
        for &value in &data {
            *value_counts.entry(value).or_insert(0) += 1;
        }
        MinVector { data, value_counts }
    }

    fn set(&mut self, index: usize, value: T) {
        // Decrease the count of the old value
        let old_value = self.data[index];
        if let Some(count) = self.value_counts.get_mut(&old_value) {
            *count -= 1;
            if *count == 0 {
                self.value_counts.remove(&old_value);
            }
        }
        // Update the value in the vector
        self.data[index] = value;

        // Increase the count of the new value
        *self.value_counts.entry(value).or_insert(0) += 1;
    }

    fn min(&self) -> T {
        *self.value_counts.keys().next().unwrap()
    }
}

impl History {
    pub fn new(world_size: usize) -> Self {
        Self {
            first_incomplete_seqs: MinVector::new(vec![Seq::default(); world_size]),
            min_incomplete_seq: Seq::default(),
            invocation_for_ref: HashMap::new(),
            invocations: HashMap::new(),
            marked_for_deletion: HashSet::new(),
            seq_lower_bound: 0.into(),
        }
    }

    #[cfg(test)]
    pub fn first_incomplete_seqs(&self) -> &[Seq] {
        self.first_incomplete_seqs.vec()
    }

    pub fn delete_invocations_for_refs(&mut self, refs: Vec<Ref>) {
        self.marked_for_deletion.extend(refs);

        self.marked_for_deletion
            .retain(|ref_| match self.invocation_for_ref.get(ref_) {
                Some(RefStatus::Invoked(seq)) => {
                    if seq < &self.min_incomplete_seq {
                        self.invocation_for_ref.remove(ref_);
                        false
                    } else {
                        true
                    }
                }
                Some(RefStatus::Errored(_)) => {
                    self.invocation_for_ref.remove(ref_);
                    false
                }
                None => true,
            });
    }

    /// Add an invocation to the history.
    pub fn add_invocation(
        &mut self,
        seq: Seq,
        uses: Vec<Ref>,
        defs: Vec<Ref>,
    ) -> Vec<(Seq, Option<Result<Serialized, Exception>>)> {
        let mut results = Vec::new();
        assert!(
            seq >= self.seq_lower_bound,
            "nonmonotonic seq: {:?}; current lower bound: {:?}",
            seq,
            self.seq_lower_bound,
        );
        self.seq_lower_bound = seq;
        let mut invocation = Invocation::new(seq, defs.clone());

        for use_ in uses {
            // The invocation for every use_ should add this seq as a user.
            match self.invocation_for_ref.get(&use_) {
                Some(RefStatus::Errored(exception)) => {
                    // We know that this invocation hasn't been completed yet, so we can
                    // directly call set_exception on it.
                    if results.is_empty() {
                        invocation.set_exception(exception.clone());
                        results.push((seq, Some(Err(exception.clone()))));
                    }
                }
                Some(RefStatus::Invoked(invoked_seq)) => {
                    if let Some(invocation) = self.invocations.get_mut(invoked_seq) {
                        invocation.add_user(seq)
                    }
                }
                None => tracing::debug!(
                    "ignoring dependency on potentially complete invocation for ref: {:?}",
                    use_
                ),
            }
        }
        for def in defs {
            self.invocation_for_ref.insert(
                def,
                match invocation.exception() {
                    Some(err) => RefStatus::Errored(err.clone()),
                    None => RefStatus::Invoked(seq.clone()),
                },
            );
        }

        self.invocations.insert(seq, invocation);

        results
    }

    /// Propagate worker error to the invocation with the given Seq. This will also propagate
    /// to all seqs that depend on this seq directly or indirectly.
    pub fn propagate_exception(
        &mut self,
        seq: Seq,
        exception: Exception,
    ) -> Vec<(Seq, Option<Result<Serialized, Exception>>)> {
        let mut results = Vec::new();
        let mut queue = vec![seq];
        let mut visited = HashSet::new();

        while let Some(seq) = queue.pop() {
            if !visited.insert(seq) {
                continue;
            }

            let Some(invocation) = self.invocations.get_mut(&seq) else {
                continue;
            };

            // Overwrite the error, so we are using the last error for this invocation to send
            // to the client.
            for def in invocation.defs.iter() {
                match self.invocation_for_ref.get(def) {
                    Some(RefStatus::Invoked(invoked_seq)) if *invoked_seq == seq => self
                        .invocation_for_ref
                        .insert(*def, RefStatus::Errored(exception.clone())),
                    _ => None,
                };
            }
            invocation.set_exception(exception.clone());
            results.push((seq, invocation.result.clone()));
            queue.extend(invocation.users.iter());
        }
        results
    }

    /// Mark the given rank as completed up to but excluding the given Seq. This will also purge history for
    /// any Seqs that are no longer relevant (completed on all ranks).
    pub fn rank_completed(
        &mut self,
        rank: usize,
        seq: Seq,
    ) -> Vec<(Seq, Option<Result<Serialized, Exception>>)> {
        self.first_incomplete_seqs.set(rank, seq);
        let prev = self.min_incomplete_seq;
        self.min_incomplete_seq = self.first_incomplete_seqs.min();

        let mut results: Vec<(Seq, Option<Result<Serialized, Exception>>)> = Vec::new();
        for i in Seq::iter_between(prev, self.min_incomplete_seq) {
            if let Some(invocation) = self.invocations.remove(&i) {
                match invocation.result {
                    Some(Err(_)) => {
                        // Retain the def history because we may need it to propagate
                        // errors in the future. We rely here on the fact that the invocation
                        // above has been marked as failed by way of failure propagation.
                        for def in &invocation.defs {
                            match self.invocation_for_ref.get(def) {
                                Some(RefStatus::Invoked(seq)) if *seq == i => {
                                    self.invocation_for_ref.remove(def)
                                }
                                _ => None,
                            };
                        }

                        // we have already reported all exceptions when they are generated.
                    }
                    e => {
                        results.push((i, e));
                    }
                }
            }
        }
        results
    }

    #[cfg(test)]
    fn get_invocation(&self, seq: Seq) -> Option<&Invocation> {
        self.invocations.get(&seq)
    }

    pub fn set_result(&mut self, seq: Seq, result: Serialized) {
        if let Some(invocation) = self.invocations.get_mut(&seq) {
            invocation.set_result(result);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use hyperactor::id;

    use super::*;

    struct InvocationUsersIterator<'a> {
        history: &'a History,
        stack: Vec<&'a Invocation>,
        visited: HashSet<Seq>,
    }

    impl<'a> Iterator for InvocationUsersIterator<'a> {
        type Item = &'a Invocation;

        fn next(&mut self) -> Option<Self::Item> {
            while let Some(invocation) = self.stack.pop() {
                if !self.visited.insert(invocation.seq) {
                    continue;
                }
                self.stack.extend(
                    invocation
                        .users
                        .iter()
                        .filter_map(|seq| self.history.invocations.get(seq)),
                );
                return Some(invocation);
            }
            None
        }
    }

    impl History {
        /// Get an iterator of Seqs that are users of (or dependent on the completion of) the given Seq.
        /// This is useful for propagating failures. This will return an empty iterator if the given Seq is
        /// not in the history. So this should be called before the invocation is marked as completed for the
        /// given rank.
        /// The Seq passed to this function will also be included in the iterator.
        pub(crate) fn iter_users_transitive(&self, seq: Seq) -> impl Iterator<Item = Seq> + '_ {
            let invocations = self
                .invocations
                .get(&seq)
                .map_or(Vec::default(), |invocation| vec![invocation]);

            InvocationUsersIterator {
                history: self,
                stack: invocations,
                visited: HashSet::new(),
            }
            .map(|invocation| invocation.seq)
        }
    }

    #[test]
    fn simple_history() {
        let mut history = History::new(2);
        history.add_invocation(0.into(), vec![], vec![Ref { id: 1 }, Ref { id: 2 }]);
        history.add_invocation(1.into(), vec![Ref { id: 1 }], vec![Ref { id: 3 }]);
        history.add_invocation(2.into(), vec![Ref { id: 3 }], vec![Ref { id: 4 }]);
        history.add_invocation(3.into(), vec![Ref { id: 3 }], vec![Ref { id: 5 }]);
        history.add_invocation(4.into(), vec![Ref { id: 3 }], vec![Ref { id: 6 }]);
        history.add_invocation(5.into(), vec![Ref { id: 4 }], vec![Ref { id: 7 }]);
        history.add_invocation(6.into(), vec![Ref { id: 4 }], vec![Ref { id: 8 }]);

        let mut res = history
            .iter_users_transitive(1.into())
            .collect::<Vec<Seq>>();
        res.sort();
        assert_eq!(
            res,
            vec![1.into(), 2.into(), 3.into(), 4.into(), 5.into(), 6.into()]
        );

        history.rank_completed(0, 2.into());
        let mut res = history
            .iter_users_transitive(1.into())
            .collect::<Vec<Seq>>();
        res.sort();
        assert_eq!(
            res,
            vec![1.into(), 2.into(), 3.into(), 4.into(), 5.into(), 6.into()]
        );

        history.rank_completed(1, 2.into());
        let res = history
            .iter_users_transitive(1.into())
            .collect::<Vec<Seq>>();
        assert_eq!(res, vec![]);

        // Test that we can still add invocations after all ranks have completed that seq.
        history.add_invocation(7.into(), vec![Ref { id: 1 }], vec![]);
    }

    #[test]
    fn delete_errored_invocations() {
        let mut history = History::new(1);
        history.add_invocation(0.into(), vec![], vec![Ref { id: 1 }, Ref { id: 2 }]);
        history.add_invocation(1.into(), vec![Ref { id: 1 }], vec![Ref { id: 3 }]);
        history.propagate_exception(
            0.into(),
            Exception::Error(
                0.into(),
                0.into(),
                WorkerError {
                    backtrace: "worker error happened".to_string(),
                    worker_actor_id: id!(test[234].testactor[6]),
                },
            ),
        );
        history.delete_invocations_for_refs(vec![Ref { id: 1 }, Ref { id: 2 }]);
        history.rank_completed(0, 1.into());
        assert_eq!(history.invocation_for_ref.len(), 1);
        history.delete_invocations_for_refs(vec![Ref { id: 3 }]);
        history.rank_completed(0, 2.into());
        assert!(history.invocation_for_ref.is_empty());
    }

    #[test]
    fn redefinitions() {
        let mut history = History::new(2);
        history.add_invocation(0.into(), vec![], vec![Ref { id: 1 }, Ref { id: 2 }]);
        history.add_invocation(1.into(), vec![Ref { id: 1 }], vec![Ref { id: 3 }]);
        history.add_invocation(2.into(), vec![Ref { id: 3 }], vec![Ref { id: 4 }]);

        let mut res = history
            .iter_users_transitive(1.into())
            .collect::<Vec<Seq>>();
        res.sort();
        assert_eq!(res, vec![1.into(), 2.into()]);

        history.add_invocation(3.into(), vec![Ref { id: 3 }], vec![Ref { id: 3 }]);
        history.add_invocation(4.into(), vec![Ref { id: 3 }], vec![Ref { id: 6 }]);
        history.add_invocation(5.into(), vec![Ref { id: 4 }], vec![Ref { id: 7 }]);
        history.add_invocation(6.into(), vec![Ref { id: 4 }], vec![Ref { id: 8 }]);

        history.rank_completed(0, 2.into());
        history.rank_completed(1, 2.into());

        let res = history
            .iter_users_transitive(3.into())
            .collect::<Vec<Seq>>();
        assert_eq!(res, vec![3.into(), 4.into()]);
    }

    #[test]
    fn min_vector() {
        // Test initialization
        let data = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let mut min_vector = MinVector::new(data.clone());

        // Test length
        assert_eq!(min_vector.len(), data.len());

        // Test initial vector
        assert_eq!(min_vector.vec(), &data);

        // Test initial minimum
        assert_eq!(min_vector.min(), 1);

        // Test get method
        for (i, &value) in data.iter().enumerate() {
            assert_eq!(min_vector.get(i), value);
        }

        // Test set method and min update
        min_vector.set(0, 0); // Change first element to 0
        assert_eq!(min_vector.get(0), 0);
        assert_eq!(min_vector.min(), 0);
        min_vector.set(1, 7); // Change second element to 7
        assert_eq!(min_vector.get(1), 7);
        assert_eq!(min_vector.min(), 0);
        min_vector.set(0, 8); // Change first element to 8
        assert_eq!(min_vector.get(0), 8);
        assert_eq!(min_vector.min(), 1); // Minimum should now be 1

        // Test setting a value that already exists
        min_vector.set(2, 5); // Change third element to 5
        assert_eq!(min_vector.get(2), 5);
        assert_eq!(min_vector.min(), 1);

        // Test setting a value to the current minimum
        min_vector.set(3, 0); // Change fourth element to 0
        assert_eq!(min_vector.get(3), 0);
        assert_eq!(min_vector.min(), 0);

        // Test setting all elements to the same value
        for i in 0..min_vector.len() {
            min_vector.set(i, 10);
        }
        assert_eq!(min_vector.min(), 10);
        assert_eq!(min_vector.vec(), &vec![10; min_vector.len()]);
    }

    #[test]
    fn failure_propagation() {
        let mut history = History::new(2);

        history.add_invocation(0.into(), vec![], vec![Ref { id: 1 }, Ref { id: 2 }]);
        history.add_invocation(1.into(), vec![Ref { id: 1 }], vec![Ref { id: 3 }]);
        history.add_invocation(
            2.into(),
            vec![Ref { id: 3 }],
            vec![Ref { id: 4 }, Ref { id: 5 }],
        );
        history.add_invocation(3.into(), vec![Ref { id: 2 }], vec![Ref { id: 6 }]);
        history.add_invocation(4.into(), vec![Ref { id: 5 }], vec![Ref { id: 6 }]);

        // No error before propagation
        for i in 1..=3 {
            assert!(
                history
                    .get_invocation(i.into())
                    .unwrap()
                    .exception()
                    .is_none()
            );
        }

        // Failure happened to invocation 1, invocations 2, 4 should be marked as failed because they
        // depend on 1 directly or indirectly.
        history.propagate_exception(
            1.into(),
            Exception::Error(
                1.into(),
                1.into(),
                WorkerError {
                    backtrace: "worker error happened".to_string(),
                    worker_actor_id: "test[234].testactor[6]".parse().unwrap(),
                },
            ),
        );

        // Error should be set for all invocations that depend on the failed invocation
        for i in [1, 2, 4] {
            assert!(
                history
                    .get_invocation(i.into())
                    .unwrap()
                    .exception()
                    .is_some()
            );
        }

        // Error should not be set for invocations that do not depend on the failed invocation
        for i in [0, 3] {
            assert!(
                history
                    .get_invocation(i.into())
                    .unwrap()
                    .exception()
                    .is_none()
            );
        }

        // A failed but completed invocation should still lead to all its
        // invocations being marked as failed even if they appear in the future.

        // Delete until 2.
        history.rank_completed(0, 2.into());
        history.rank_completed(1, 2.into());

        for i in [3, 4, 5, 6] {
            assert_matches!(
                history.invocation_for_ref.get(&i.into()),
                Some(RefStatus::Errored(_)),
            );
            // Invocation should start from 5, so i+2
            history.add_invocation((i + 2).into(), vec![Ref { id: i }], vec![Ref { id: 7 }]);
            assert!(
                history
                    .get_invocation((i + 2).into())
                    .unwrap()
                    .exception()
                    .is_some()
            );
        }

        // Test if you can fill a valid result on an errored invocation 2.
        history.set_result(2.into(), Serialized::serialize(&"2".to_string()).unwrap());
        // check that seq 2 is still errored
        assert!(
            history
                .get_invocation((2).into())
                .unwrap()
                .exception()
                .is_some()
        );
        assert!(
            history
                .get_invocation((2).into())
                .unwrap()
                .value()
                .is_none()
        );
    }
}
