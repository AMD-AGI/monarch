# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import asyncio
import functools
import importlib.resources
import os
import re
import signal
import subprocess
import sys
from typing import cast, List, Tuple
from unittest.mock import AsyncMock, patch

import monarch
import monarch.actor as actor

import pytest

import torch
from monarch._src.actor.actor_mesh import Actor, ActorError, current_rank, IN_PAR
from monarch._src.actor.debugger import (
    Attach,
    Cast,
    Continue,
    DebugCommand,
    DebugController,
    DebugSession,
    DebugSessionInfo,
    DebugSessions,
    DebugStdIO,
    Help,
    ListCommand,
    Quit,
)
from monarch._src.actor.endpoint import endpoint
from monarch._src.actor.proc_mesh import proc_mesh

from pyre_extensions import none_throws

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def _debug_port():
    for i in range(100):
        yield {
            "MONARCH_DEBUG_SERVER_PORT": f"270{i:02d}",
        }


debug_port = _debug_port()


def isolate_in_subprocess(test_fn=None, *, env=None):
    if test_fn is None:
        return functools.partial(isolate_in_subprocess, env=env)

    if env is None:
        env = {}

    def sync_test_fn():
        asyncio.run(test_fn())

    sync_test_fn_name = f"sync_{test_fn.__name__}"
    setattr(sys.modules[__name__], sync_test_fn_name, sync_test_fn)

    env.update(os.environ.copy())

    def wrapper():
        if IN_PAR:
            assert (
                subprocess.call(
                    [
                        str(
                            importlib.resources.files("monarch.python.tests").joinpath(
                                "run_test_bin"
                            )
                        ),
                        sync_test_fn_name,
                    ],
                    env=env,
                )
                == 0
            )
        else:
            assert (
                subprocess.call(
                    [
                        sys.executable,
                        "-c",
                        f"import tests.test_debugger; tests.test_debugger.{sync_test_fn_name}()",
                    ],
                    env=env,
                )
                == 0
            )

    return wrapper


def run_test_from_name():
    getattr(sys.modules[__name__], sys.argv[1])()


debug_cli_bin = (
    str(importlib.resources.files("monarch.python.tests").joinpath("debug_cli_bin"))
    if IN_PAR
    else ""
)


def _bad_rank():
    raise ValueError("bad rank")


def _debugee_actor_internal(rank):
    if rank == 0:
        breakpoint()  # noqa
        rank += 1
        rank += 1
        return rank
    elif rank == 1:
        breakpoint()  # noqa
        rank += 2
        rank += 2
        return rank
    elif rank == 2:
        breakpoint()  # noqa
        rank += 3
        rank += 3
        _bad_rank()
    elif rank == 3:
        breakpoint()  # noqa
        rank += 4
        rank += 4
        return rank


class DebugeeActor(Actor):
    @endpoint
    async def to_debug(self):
        rank = current_rank().rank
        return _debugee_actor_internal(rank)


class DebugControllerForTesting(DebugController):
    def __init__(self):
        super().__init__()
        self._debug_io = DebugStdIO()

    @endpoint
    async def blocking_enter(self):
        async with self._task_lock:
            assert self._task is None
            await self._enter()


async def _wait_for_breakpoints(
    debug_controller, n_breakpoints
) -> List[DebugSessionInfo]:
    breakpoints: List[DebugSessionInfo] = []
    for i in range(20):
        breakpoints = await debug_controller.list.call_one(print_output=False)
        if len(breakpoints) == n_breakpoints:
            break
        await asyncio.sleep(1)
        if i == 20:
            raise RuntimeError("timed out waiting for breakpoints")
    return breakpoints


# We have to run this test in a separate process because there is only one
# debug controller per process, and we don't want this to interfere with
# the other two tests that access the debug controller.
@isolate_in_subprocess(env=next(debug_port))
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Not enough GPUs, this test requires at least 2 GPUs",
)
@pytest.mark.timeout(180)
async def test_debug() -> None:
    input_mock = AsyncMock()
    input_mock.side_effect = [
        "attach debugee 1",
        "n",
        "n",
        "n",
        "n",
        "detach",
        "attach debugee 1",
        "detach",
        "quit",
        "cast debugee ranks(0,3) n",
        "cast debugee ranks(0,3) n",
        # Attaching to 0 and 3 ensures that when we call "list"
        # the next time, their function/lineno info will be
        # up-to-date.
        "attach debugee 0",
        "detach",
        "attach debugee 3",
        "detach",
        "quit",
        "attach debugee 2",
        "c",
        "detach",
        "quit",
        "attach debugee 2",
        "bt",
        "c",
        "quit",
        "continue",
        "quit",
    ]

    outputs = []

    def _patch_output(msg):
        nonlocal outputs
        outputs.append(msg)

    output_mock = AsyncMock()
    output_mock.side_effect = _patch_output

    with patch("monarch._src.actor.debugger.DebugStdIO.input", new=input_mock), patch(
        "monarch._src.actor.debugger.DebugStdIO.output", new=output_mock
    ):
        proc = proc_mesh(hosts=2, gpus=2)
        debugee = await proc.spawn("debugee", DebugeeActor)
        debug_controller = actor.get_or_spawn_controller(
            "debug_controller", DebugControllerForTesting
        ).get()

        fut = debugee.to_debug.call()
        await debug_controller.wait_pending_session.call_one()
        breakpoints = await _wait_for_breakpoints(debug_controller, 4)

        initial_linenos = {}
        for i in range(len(breakpoints)):
            info = breakpoints[i]
            initial_linenos[info.rank] = info.lineno
            assert info.rank == i
            assert info.coords == {"hosts": info.rank // 2, "gpus": info.rank % 2}
            assert info.function == "test_debugger._debugee_actor_internal"
            assert info.lineno == cast(int, breakpoints[0].lineno) + 5 * info.rank

        await debug_controller.blocking_enter.call_one()

        # Check that when detaching and re-attaching to a session, the last portion of the output is repeated
        expected_last_output = [
            r"--Return--",
            r"\n",
            r"> (/.*/)+test_debugger.py\(\d+\)to_debug\(\)->5\n-> return _debugee_actor_internal\(rank\)",
            r"\n",
            r"\(Pdb\) ",
        ]
        output_len = len(expected_last_output)
        rev_outputs = outputs[::-1]
        last_return = rev_outputs.index("--Return--")
        second_to_last_return = rev_outputs.index("--Return--", last_return + 1)
        last_return = len(rev_outputs) - last_return - 1
        second_to_last_return = len(rev_outputs) - second_to_last_return - 1
        assert (
            outputs[second_to_last_return : second_to_last_return + output_len]  # noqa
            == outputs[last_return : last_return + output_len]  # noqa
        )
        for real_output, expected_output in zip(
            outputs[last_return : last_return + output_len],  # noqa
            expected_last_output,
        ):
            assert re.match(expected_output, real_output) is not None

        breakpoints = await debug_controller.list.call_one(print_output=False)
        for i in range(len(breakpoints)):
            if i == 1:
                assert breakpoints[i].function == "test_debugger.to_debug"
            else:
                assert (
                    breakpoints[i].function == "test_debugger._debugee_actor_internal"
                )
                assert breakpoints[i].lineno == initial_linenos[i]

        await debug_controller.blocking_enter.call_one()

        breakpoints = await debug_controller.list.call_one(print_output=False)
        for i in range(len(breakpoints)):
            if i == 1:
                assert breakpoints[i].function == "test_debugger.to_debug"
            elif i in (0, 3):
                assert (
                    breakpoints[i].function == "test_debugger._debugee_actor_internal"
                )
                assert breakpoints[i].lineno == initial_linenos[i] + 2
            else:
                assert (
                    breakpoints[i].function == "test_debugger._debugee_actor_internal"
                )
                assert breakpoints[i].lineno == initial_linenos[i]

        await debug_controller.blocking_enter.call_one()

        breakpoints = await debug_controller.list.call_one(print_output=False)
        assert len(breakpoints) == 4
        # Expect post-mortem debugging for rank 2
        assert breakpoints[2].function == "test_debugger._bad_rank"

        await debug_controller.blocking_enter.call_one()

        expected_last_output = [
            r"\s*(/.*/)+test_debugger.py\(\d+\)_debugee_actor_internal\(\)\n-> _bad_rank\(\)",
            r"\n",
            r'> (/.*/)+test_debugger.py\(\d+\)_bad_rank\(\)\n-> raise ValueError\("bad rank"\)',
            r"\n",
            r"\(Pdb\) ",
        ]

        rev_outputs = outputs[::-1]
        output_index = len(outputs) - (
            rev_outputs.index("(Pdb) ") + len(expected_last_output)
        )

        for output, expected_output in zip(
            outputs[output_index : output_index + len(expected_last_output)],  # noqa
            expected_last_output,
        ):
            assert re.match(expected_output, output) is not None

        breakpoints = await debug_controller.list.call_one(print_output=False)
        assert len(breakpoints) == 3
        for i, rank in enumerate((0, 1, 3)):
            assert breakpoints[i].rank == rank

        await debug_controller.blocking_enter.call_one()
        breakpoints = await debug_controller.list.call_one(print_output=False)
        assert len(breakpoints) == 0

        with pytest.raises(
            monarch._src.actor.actor_mesh.ActorError, match="ValueError: bad rank"
        ):
            await fut


# See earlier comment
@isolate_in_subprocess(env=next(debug_port))
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Not enough GPUs, this test requires at least 2 GPUs",
)
@pytest.mark.timeout(180)
async def test_debug_multi_actor() -> None:
    input_mock = AsyncMock()
    input_mock.side_effect = [
        "attach debugee_2 2",
        "n",
        "detach",
        "attach debugee_1 1",
        "n",
        "detach",
        "quit",
        "cast debugee_1 ranks(:) c",
        "cast debugee_2 ranks(:) c",
        "attach debugee_2 2",
        "c",
        "quit",
        "continue",
        "quit",
    ]

    with patch("monarch._src.actor.debugger.DebugStdIO.input", side_effect=input_mock):
        proc = await proc_mesh(hosts=2, gpus=2)
        debugee_1 = await proc.spawn("debugee_1", DebugeeActor)
        debugee_2 = await proc.spawn("debugee_2", DebugeeActor)
        debug_controller = actor.get_or_spawn_controller(
            "debug_controller", DebugControllerForTesting
        ).get()

        fut_1 = debugee_1.to_debug.call()
        fut_2 = debugee_2.to_debug.call()
        await debug_controller.wait_pending_session.call_one()

        breakpoints = await _wait_for_breakpoints(debug_controller, 8)

        initial_linenos = {}
        for i in range(len(breakpoints)):
            info = breakpoints[i]
            initial_linenos[info.rank] = info.lineno
            assert info.rank == i % 4
            assert info.actor_name == "debugee_1" if i < 4 else "debugee_2"
            assert info.coords == {"hosts": info.rank // 2, "gpus": info.rank % 2}
            assert info.function == "test_debugger._debugee_actor_internal"
            assert info.lineno == cast(int, breakpoints[0].lineno) + 5 * info.rank

        await debug_controller.blocking_enter.call_one()

        breakpoints = await _wait_for_breakpoints(debug_controller, 8)
        for i in range(len(breakpoints)):
            if i == 1:
                assert breakpoints[i].actor_name == "debugee_1"
                assert breakpoints[i].rank == 1
                assert breakpoints[i].lineno == initial_linenos[breakpoints[i].rank] + 1
            elif i == 6:
                assert breakpoints[i].actor_name == "debugee_2"
                assert breakpoints[i].rank == 2
                assert breakpoints[i].lineno == initial_linenos[breakpoints[i].rank] + 1
            else:
                assert (
                    breakpoints[i].actor_name == "debugee_1" if i < 4 else "debugee_2"
                )
                assert breakpoints[i].rank == i % 4
                assert breakpoints[i].lineno == initial_linenos[breakpoints[i].rank]

        await debug_controller.blocking_enter.call_one()

        breakpoints = await _wait_for_breakpoints(debug_controller, 1)
        with pytest.raises(ActorError, match="ValueError: bad rank"):
            await fut_2
        assert breakpoints[0].actor_name == "debugee_1"
        assert breakpoints[0].rank == 2
        assert breakpoints[0].function == "test_debugger._bad_rank"

        await debug_controller.blocking_enter.call_one()

        breakpoints = await _wait_for_breakpoints(debug_controller, 0)
        with pytest.raises(ActorError, match="ValueError: bad rank"):
            await fut_1


async def test_debug_sessions_insert_get_remove() -> None:
    mock_sessions = []
    for actor_name in ("actor_a", "actor_b"):
        for rank in range(2):
            mock_session = DebugSession(rank, {}, "", actor_name)
            mock_sessions.append(mock_session)

    debug_sessions = DebugSessions()

    with pytest.raises(ValueError, match="No debug sessions for actor actor_a"):
        debug_sessions.get("actor_a", 0)
    debug_sessions.insert(mock_sessions[0])
    assert debug_sessions.get("actor_a", 0) is mock_sessions[0]
    assert ("actor_a", 0) in debug_sessions
    with pytest.raises(
        ValueError, match="Debug session for rank 0 already exists for actor actor_a"
    ):
        debug_sessions.insert(mock_sessions[0])

    with pytest.raises(
        ValueError, match="No debug session for rank 1 for actor actor_a"
    ):
        debug_sessions.get("actor_a", 1)
    debug_sessions.insert(mock_sessions[1])
    assert debug_sessions.get("actor_a", 1) is mock_sessions[1]
    assert ("actor_a", 1) in debug_sessions
    with pytest.raises(
        ValueError, match="Debug session for rank 1 already exists for actor actor_a"
    ):
        debug_sessions.insert(mock_sessions[1])

    with pytest.raises(ValueError, match="No debug sessions for actor actor_b"):
        debug_sessions.get("actor_b", 0)
    debug_sessions.insert(mock_sessions[2])
    assert debug_sessions.get("actor_b", 0) is mock_sessions[2]
    assert ("actor_b", 0) in debug_sessions
    with pytest.raises(
        ValueError, match="Debug session for rank 0 already exists for actor actor_b"
    ):
        debug_sessions.insert(mock_sessions[2])

    with pytest.raises(
        ValueError, match="No debug session for rank 1 for actor actor_b"
    ):
        debug_sessions.get("actor_b", 1)
    debug_sessions.insert(mock_sessions[3])
    assert debug_sessions.get("actor_b", 1) is mock_sessions[3]
    assert ("actor_b", 1) in debug_sessions
    with pytest.raises(
        ValueError, match="Debug session for rank 1 already exists for actor actor_b"
    ):
        debug_sessions.insert(mock_sessions[3])

    assert len(debug_sessions) == 4

    assert debug_sessions.remove("actor_a", 0) is mock_sessions[0]
    assert len(debug_sessions) == 3
    assert ("actor_a", 0) not in debug_sessions
    with pytest.raises(
        ValueError, match="No debug session for rank 0 for actor actor_a"
    ):
        debug_sessions.remove("actor_a", 0)

    assert debug_sessions.remove("actor_a", 1) is mock_sessions[1]
    assert len(debug_sessions) == 2
    assert ("actor_a", 1) not in debug_sessions
    with pytest.raises(ValueError, match="No debug sessions for actor actor_a"):
        debug_sessions.remove("actor_a", 1)

    assert debug_sessions.remove("actor_b", 0) is mock_sessions[2]
    assert len(debug_sessions) == 1
    assert ("actor_b", 0) not in debug_sessions
    with pytest.raises(
        ValueError, match="No debug session for rank 0 for actor actor_b"
    ):
        debug_sessions.remove("actor_b", 0)

    assert debug_sessions.remove("actor_b", 1) is mock_sessions[3]
    assert len(debug_sessions) == 0
    assert ("actor_b", 1) not in debug_sessions
    with pytest.raises(ValueError, match="No debug sessions for actor actor_b"):
        debug_sessions.remove("actor_b", 1)


async def test_debug_sessions_iter() -> None:
    debug_sessions = DebugSessions()
    mock_sessions = []

    for actor_name in ("actor_a", "actor_b"):
        for host in range(3):
            for gpu in range(8):
                rank = host * 8 + gpu
                mock_session = DebugSession(
                    rank, {"hosts": host, "gpus": gpu}, "", actor_name
                )
                mock_sessions.append(mock_session)
                debug_sessions.insert(mock_session)

    # Single rank
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = list(debug_sessions.iter((actor_name, 2)))
        assert len(sessions) == 1
        assert sessions[0] is mock_sessions[i * 24 + 2]

    # List of ranks
    ranks = [1, 3, 5]
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = sorted(
            debug_sessions.iter((actor_name, ranks)), key=lambda s: s.get_info()
        )
        assert len(sessions) == 3
        for j in range(3):
            assert sessions[j] is mock_sessions[i * 24 + ranks[j]]

    # Range of ranks
    ranks = range(2, 24, 3)
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = sorted(
            debug_sessions.iter((actor_name, ranks)), key=lambda s: s.get_info()
        )
        ranks = list(ranks)
        assert len(sessions) == len(ranks)
        for j in range(len(ranks)):
            assert sessions[j] is mock_sessions[i * 24 + ranks[j]]

    # All ranks
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = sorted(
            debug_sessions.iter((actor_name, None)), key=lambda s: s.get_info()
        )
        assert len(sessions) == 24
        for j in range(24):
            assert sessions[j] is mock_sessions[i * 24 + j]

    # All ranks, all actors
    sessions = sorted(debug_sessions.iter(None), key=lambda s: s.get_info())
    assert len(sessions) == 48
    for i in range(48):
        assert sessions[i] is mock_sessions[i]

    # Dimension filtering with a single value
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = sorted(
            debug_sessions.iter((actor_name, {"hosts": 1})), key=lambda s: s.get_info()
        )
        assert len(sessions) == 8
        for j in range(8):
            assert sessions[j] is mock_sessions[i * 24 + 8 + j]

    # Dimension filtering with a list
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = sorted(
            debug_sessions.iter((actor_name, {"hosts": [0, 2]})),
            key=lambda s: s.get_info(),
        )
        assert len(sessions) == 16
        j = 0
        for host in (0, 2):
            for gpu in range(8):
                assert sessions[j] is mock_sessions[i * 24 + host * 8 + gpu]
                j += 1

    # Dimension filtering with a range
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = sorted(
            debug_sessions.iter((actor_name, {"gpus": range(5, 8)})),
            key=lambda s: s.get_info(),
        )
        assert len(sessions) == 9
        j = 0
        for host in range(3):
            for gpu in range(5, 8):
                assert sessions[j] is mock_sessions[i * 24 + host * 8 + gpu]
                j += 1

    # Multiple dimension filters
    for i, actor_name in enumerate(("actor_a", "actor_b")):
        sessions = sorted(
            debug_sessions.iter(
                (actor_name, {"hosts": [1, 3], "gpus": range(0, sys.maxsize, 3)})
            ),
            key=lambda s: s.get_info(),
        )
        assert len(sessions) == 3
        j = 0
        for gpu in range(0, 8, 3):
            assert sessions[j] is mock_sessions[i * 24 + 8 + gpu]
            j += 1

    # Non-existent dimension
    for actor_name in ("actor_a", "actor_b"):
        sessions = sorted(
            debug_sessions.iter((actor_name, {"hosts": 0, "gpus": 0, "foo": 0})),
            key=lambda s: s.get_info(),
        )
        assert len(sessions) == 0


@pytest.mark.parametrize(
    ["user_input", "expected_output"],
    [
        ("attach debugee 1", Attach("debugee", 1)),
        ("a my_awesome_actor-123_DBG 100", Attach("my_awesome_actor-123_DBG", 100)),
        ("list", ListCommand()),
        ("l", ListCommand()),
        ("help", Help()),
        ("h", Help()),
        ("quit", Quit()),
        ("q", Quit()),
        ("continue", Continue()),
        ("c", Continue()),
        (
            "cast debugee ranks(123) b 25",
            Cast(actor_name="debugee", ranks=123, command="b 25"),
        ),
        (
            "cast my_awesome_actor ranks(12,34,56) b 25",
            Cast(actor_name="my_awesome_actor", ranks=[12, 34, 56], command="b 25"),
        ),
        (
            "cast debugee ranks(:) b 25",
            Cast(actor_name="debugee", ranks=range(0, sys.maxsize), command="b 25"),
        ),
        (
            "cast debugee ranks(:123) b 25",
            Cast(actor_name="debugee", ranks=range(0, 123), command="b 25"),
        ),
        (
            "cast debugee ranks(123:) b 25",
            Cast(actor_name="debugee", ranks=range(123, sys.maxsize), command="b 25"),
        ),
        (
            "cast debugee ranks(123:456) b 25",
            Cast(actor_name="debugee", ranks=range(123, 456), command="b 25"),
        ),
        (
            "cast debugee ranks(::) b 25",
            Cast(actor_name="debugee", ranks=range(0, sys.maxsize), command="b 25"),
        ),
        (
            "cast debugee ranks(::123) b 25",
            Cast(
                actor_name="debugee", ranks=range(0, sys.maxsize, 123), command="b 25"
            ),
        ),
        (
            "cast debugee ranks(123::) b 25",
            Cast(actor_name="debugee", ranks=range(123, sys.maxsize), command="b 25"),
        ),
        (
            "cast debugee ranks(:123:) b 25",
            Cast(actor_name="debugee", ranks=range(0, 123), command="b 25"),
        ),
        (
            "cast debugee ranks(:456:123) b 25",
            Cast(actor_name="debugee", ranks=range(0, 456, 123), command="b 25"),
        ),
        (
            "cast debugee ranks(456::123) b 25",
            Cast(
                actor_name="debugee", ranks=range(456, sys.maxsize, 123), command="b 25"
            ),
        ),
        (
            "cast debugee ranks(123:456:) b 25",
            Cast(actor_name="debugee", ranks=range(123, 456), command="b 25"),
        ),
        (
            "cast debugee ranks(456:789:123) b 25",
            Cast(actor_name="debugee", ranks=range(456, 789, 123), command="b 25"),
        ),
        (
            "cast debugee ranks(dim1=123) up 2",
            Cast(actor_name="debugee", ranks={"dim1": 123}, command="up 2"),
        ),
        (
            "cast debugee ranks(dim1=123, dim2=(12,34,56), dim3=15::2) up 2",
            Cast(
                actor_name="debugee",
                ranks={
                    "dim1": 123,
                    "dim2": [12, 34, 56],
                    "dim3": range(15, sys.maxsize, 2),
                },
                command="up 2",
            ),
        ),
    ],
)
async def test_debug_command_parser_valid_inputs(user_input, expected_output):
    assert await DebugCommand.parse(DebugStdIO(), user_input) == expected_output


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "a",
        "attach",
        "a actor",
        "attach actor",
        "attacha actor 1" "attch actor 1",
        "attach actor 1abc",
        "attach actor 1 a",
        "cast ranks(123) b 25",
        "cast   ranks(123) b 25",
        "castactor ranks(123) b 25",
        "cast actor rnks(123) b 25",
        "cast actor ranks() b 25",
        "cast actor ranks(1ab) b 25",
        "cast actor ranks(1,a,3) b 25",
        "cast actor ranks(a:2:4) b 25",
        "cast actor ranks(1,2,3",
        "cast actor ranks(1,2,3)) b 25",
        "cast actor ranks(1,) b 25",
        "cast actor ranks(1,2,) b 25",
        "cast actor ranks(,1,2) b 25",
        "cast actor ranks(1,,2) b 25",
        "cast actor ranks(:::) b 25",
        "cast actor ranks(:123::) b 25",
        "cast actor ranks(1:2:3,4) b 25",
        "cast actor ranks(dim1=) b 25",
        "cast actor ranks(dim1=123, dim2=) b 25",
        "cast actor ranks(dim1=123, dim2=(12,34,56) b 25",
        "cast actor ranks(dim1=123, dim2=(,12,34,56) b 25",
        "cast actor ranks(dim1=123, dim2=(12,,34,56) b 25",
        "cast actor ranks(dim1=123, dim2=(12,34,56), dim3=15::2 b 25",
        "cast actor ranks(dim1=123,) b 25",
    ],
)
async def test_debug_command_parser_invalid_inputs(invalid_input):
    assert await DebugCommand.parse(DebugStdIO(), invalid_input) is None


# See earlier comment
@isolate_in_subprocess(env={"MONARCH_DEBUG_CLI_BIN": debug_cli_bin, **next(debug_port)})
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Not enough GPUs, this test requires at least 2 GPUs",
)
@pytest.mark.timeout(180)
async def test_debug_cli():
    proc = proc_mesh(hosts=2, gpus=2)
    debugee = await proc.spawn("debugee", DebugeeActor)
    debug_controller = actor.debug_controller()

    fut = debugee.to_debug.call()
    breakpoints = await _wait_for_breakpoints(debug_controller, 4)

    initial_linenos = {}
    for i in range(len(breakpoints)):
        info = breakpoints[i]
        initial_linenos[info.rank] = info.lineno
        assert info.rank == i
        assert info.coords == {"hosts": info.rank // 2, "gpus": info.rank % 2}
        assert info.function == "test_debugger._debugee_actor_internal"
        assert info.lineno == cast(int, breakpoints[0].lineno) + 5 * info.rank

    async def create_debug_cli_proc() -> (
        Tuple[asyncio.subprocess.Process, asyncio.StreamWriter, asyncio.StreamReader]
    ):
        if IN_PAR:
            cmd = [os.environ["MONARCH_DEBUG_CLI_BIN"]]
        else:
            cmd = [sys.executable, "-m", "monarch.debug_cli"]
        debug_cli_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        debug_cli_stdin = none_throws(debug_cli_proc.stdin)
        debug_cli_stdout = none_throws(debug_cli_proc.stdout)
        return debug_cli_proc, debug_cli_stdin, debug_cli_stdout

    (
        debug_cli_proc,
        debug_cli_stdin,
        debug_cli_stdout,
    ) = await create_debug_cli_proc()

    debug_cli_stdin.writelines(
        [
            b"attach debugee 1\n",
            b"n\n",
            b"n\n",
            b"n\n",
            b"n\n",
            b"detach\n",
            b"attach debugee 1\n",
            b"print('test separator')\n",
            b"detach\n",
        ]
    )
    await debug_cli_stdin.drain()

    # Check that when detaching and re-attaching to a session, the last portion of the output is repeated
    expected_last_output = (
        r"--Return--\n"
        r"> (?:/.*/)+test_debugger.py\(\d+\)to_debug\(\)->5\n"
        r"-> return _debugee_actor_internal\(rank\)\n"
        r"\(Pdb\) "
    )

    outputs = (await debug_cli_stdout.readuntil(b"test separator")).decode()
    assert len(re.findall(expected_last_output, outputs)) == 2
    assert outputs[0] == outputs[1]

    breakpoints = await debug_controller.list.call_one(print_output=False)
    for i in range(len(breakpoints)):
        if i == 1:
            assert breakpoints[i].function == "test_debugger.to_debug"
        else:
            assert breakpoints[i].function == "test_debugger._debugee_actor_internal"
            assert breakpoints[i].lineno == initial_linenos[i]

    debug_cli_stdin.write(b"quit\n")
    await debug_cli_stdin.drain()
    assert await debug_cli_proc.wait() == 0

    (
        debug_cli_proc,
        debug_cli_stdin,
        debug_cli_stdout,
    ) = await create_debug_cli_proc()

    debug_cli_stdin.writelines(
        [
            b"cast debugee ranks(0,3) n\n",
            b"cast debugee ranks(0,3) n\n",
            # Attaching to 0 and 3 ensures that when we call "list"
            # the next time, their function/lineno info will be
            # up-to-date.
            b"attach debugee 0\n",
            b"detach\n",
            b"attach debugee 3\n",
            b"detach\n",
        ]
    )
    await debug_cli_stdin.drain()

    # Make sure we have run all the commands before killing the CLI, otherwise
    # the commands may not actually be sent to the debug controller.
    await debug_cli_stdout.readuntil(b"Detached from debug session for debugee 3")
    # Even if we kill the proc using a signal, we should be able to reconnect
    # without issue.
    debug_cli_proc.send_signal(signal.SIGINT)
    assert await debug_cli_proc.wait() != 0

    breakpoints = await debug_controller.list.call_one(print_output=False)
    for i in range(len(breakpoints)):
        if i == 1:
            assert breakpoints[i].function == "test_debugger.to_debug"
        elif i in (0, 3):
            assert breakpoints[i].function == "test_debugger._debugee_actor_internal"
            assert breakpoints[i].lineno == initial_linenos[i] + 2
        else:
            assert breakpoints[i].function == "test_debugger._debugee_actor_internal"
            assert breakpoints[i].lineno == initial_linenos[i]

    (
        debug_cli_proc,
        debug_cli_stdin,
        debug_cli_stdout,
    ) = await create_debug_cli_proc()

    debug_cli_stdin.writelines([b"attach debugee 2\n", b"c\n"])
    await debug_cli_stdin.drain()

    # Make sure we have run all the commands before killing the CLI, otherwise
    # the commands may not actually be sent to the debug controller.
    await debug_cli_stdout.readuntil(b"raise ValueError")
    # Even if we kill the proc using a signal while the debugger is attached to
    # a specific rank, we should be able to reconnect to that rank later without
    # issue.
    debug_cli_proc.send_signal(signal.SIGINT)
    assert await debug_cli_proc.wait() != 0

    breakpoints = await debug_controller.list.call_one(print_output=False)
    assert len(breakpoints) == 4
    # Expect post-mortem debugging for rank 2
    assert breakpoints[2].function == "test_debugger._bad_rank"

    (
        debug_cli_proc,
        debug_cli_stdin,
        debug_cli_stdout,
    ) = await create_debug_cli_proc()

    debug_cli_stdin.writelines([b"attach debugee 2\n", b"bt\n", b"c\n"])
    await debug_cli_stdin.drain()

    expected_output = (
        r"(?:/.*/)+test_debugger.py\(\d+\)_debugee_actor_internal\(\)\n-> _bad_rank\(\)\n"
        r'> (?:/.*/)+test_debugger.py\(\d+\)_bad_rank\(\)\n-> raise ValueError\("bad rank"\)\n'
        r"\(Pdb\)"
    )

    output = (
        await debug_cli_stdout.readuntil(b"Detached from debug session for debugee 2")
    ).decode()
    assert len(re.findall(expected_output, output)) == 1

    debug_cli_stdin.writelines([b"quit\n"])
    await debug_cli_stdin.drain()
    assert await debug_cli_proc.wait() == 0

    breakpoints = await debug_controller.list.call_one(print_output=False)
    assert len(breakpoints) == 3
    for i, rank in enumerate((0, 1, 3)):
        assert breakpoints[i].rank == rank

    debug_cli_proc, debug_cli_stdin, _ = await create_debug_cli_proc()
    debug_cli_stdin.writelines([b"continue\n", b"quit\n"])
    await debug_cli_stdin.drain()
    assert await debug_cli_proc.wait() == 0

    breakpoints = await debug_controller.list.call_one(print_output=False)
    assert len(breakpoints) == 0

    with pytest.raises(
        monarch._src.actor.actor_mesh.ActorError, match="ValueError: bad rank"
    ):
        await fut
