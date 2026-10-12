"""Generated control-flow checks: temporary sysfs and mocked process/signal I/O.

No board, real sysfs, real subprocess, model, compiler or SDK is used. Run with
python3 -B test_npu_frequency_test.py. The controller is imported without running
its CLI; only main() inside the complete fake environment is exercised.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name('npu_frequency_test.py')
spec = importlib.util.spec_from_file_location('r59_frequency_control_under_test', SCRIPT)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class FakeChild:
    pid = 987654321

    def __init__(self, code):
        self.returncode = code

    def poll(self):
        return self.returncode


class FakeBoard:
    def __init__(self, root, scenario):
        self.root, self.scenario = root, scenario
        self.dev, self.therm = root / 'devfreq', root / 'thermal'
        self.dev.mkdir(); self.therm.mkdir()
        self.original = {'governor': 'userspace', 'cur_freq': '300000000',
            'min_freq': '300000000', 'max_freq': '1000000000',
            'userspace/set_freq': 'undefined',
            'available_frequencies': '300000000 600000000 1000000000'}
        for name, value in self.original.items():
            path = self.dev / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        self.sensors = []
        for index in range(7):
            zone = self.therm / f'thermal_zone{index}'
            zone.mkdir(); (zone / 'type').write_text(f'sensor-{index}')
            sensor = zone / 'temp'; sensor.write_text('42000')
            self.sensors.append(sensor)
        self.compatible = root / 'compatible'
        self.compatible.write_bytes(b'rockchip,rk3588\x00')
        self.launcher = root / 'launcher.sh'
        self.launcher.write_text('# Test fixture only; never executed.\n')
        self.child = FakeChild(None if scenario in ('thermal', 'read-error', 'timeout')
                               else 7 if scenario == 'nonzero' else 0)
        self.group_alive = self.child.returncode is None or scenario == 'leader-exited'
        self.now, self.cur_reads = 0., 0
        self.kill_calls, self.writes, self.signal_calls, self.events = [], [], [], []
        self.real_read, self.real_write = Path.read_text, Path.write_text

    def path(self, value):
        return self.compatible if str(value) == '/proc/device-tree/compatible' else Path(value)

    def read_text(self, path, *args, **kwargs):
        if path == self.dev / 'cur_freq':
            self.cur_reads += 1
            if self.cur_reads == 3:
                if self.scenario == 'read-error':
                    raise OSError('Generated monitoring read failure')
                if self.scenario == 'thermal':
                    self.real_write(self.sensors[0], '75000')
                if self.scenario == 'missing-sensor':
                    self.sensors[-1].unlink()
        return self.real_read(path, *args, **kwargs)

    def write_text(self, path, text, *args, **kwargs):
        if not path.resolve().is_relative_to(self.root):
            raise AssertionError(f'Write escaped the fake filesystem: {path}')
        if path == self.dev / 'userspace/set_freq':
            self.writes.append(str(text)); self.events.append(('frequency', str(text)))
            if self.scenario == 'restore-error' and str(text) == '300000000':
                raise OSError('Generated restore write failure')
            result = self.real_write(path, text, *args, **kwargs)
            self.real_write(self.dev / 'cur_freq', str(text))
            return result
        return self.real_write(path, text, *args, **kwargs)

    def popen(self, argv, **kwargs):
        if argv != ['sh', str(self.launcher)] or kwargs.get('start_new_session') is not True:
            raise AssertionError('Unexpected fake worker launch contract')
        return self.child

    def killpg(self, pgid, sig):
        if pgid != self.child.pid:
            raise AssertionError('Unexpected process group')
        self.kill_calls.append(sig)
        if not self.group_alive:
            raise ProcessLookupError('Generated empty process group')
        if sig == 0:
            return
        self.events.append(('signal', sig))
        if sig == signal.SIGTERM and self.scenario == 'leader-exited':
            return  # A descendant survives TERM after its shell leader exited.
        if sig not in (signal.SIGTERM, signal.SIGKILL):
            raise AssertionError('Unexpected signal')
        self.group_alive = False
        if self.child.returncode is None:
            self.child.returncode = -int(sig)

    def sleep(self, seconds):
        self.now += seconds
        if self.now > 11:
            raise AssertionError('Generated control flow exceeded its bounded wait')

    def signal(self, sig, handler):
        self.signal_calls.append((sig, handler))
        return signal.SIG_DFL  # No real signal handler is touched.


class FrequencySafetyTests(unittest.TestCase):
    def run_case(self, scenario):
        with tempfile.TemporaryDirectory(prefix='vggt-frequency-fake-') as temporary:
            board = FakeBoard(Path(temporary), scenario)
            output = board.root / 'result'
            argv = [str(SCRIPT), '--execute-frequency-test', '--launcher', str(board.launcher),
                    '--output', str(output), '--temperature-limit-c', '75',
                    '--timeout-seconds', '.25' if scenario == 'timeout' else '180']
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(controller, 'DEV', board.dev))
                stack.enter_context(patch.object(controller, 'THERM', board.therm))
                stack.enter_context(patch.object(controller, 'Path', board.path))
                stack.enter_context(patch.object(Path, 'read_text',
                    lambda path, *args, **kwargs: board.read_text(path, *args, **kwargs)))
                stack.enter_context(patch.object(Path, 'write_text',
                    lambda path, text, *args, **kwargs: board.write_text(path, text, *args, **kwargs)))
                stack.enter_context(patch.object(controller.subprocess, 'Popen', board.popen))
                stack.enter_context(patch.object(controller.os, 'killpg', board.killpg))
                stack.enter_context(patch.object(controller.time, 'monotonic', lambda: board.now))
                stack.enter_context(patch.object(controller.time, 'sleep', board.sleep))
                stack.enter_context(patch.object(controller.signal, 'signal', board.signal))
                stack.enter_context(patch.object(sys, 'argv', argv))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code = controller.main()
            report = json.loads((output / 'control.json').read_text())
            board.final = {name: (board.dev / name).read_text().strip() for name in board.original}
        return code, report, board

    def assert_restored(self, report, board):
        self.assertIs(report['frequency_governor_limits_restored'], True)
        self.assertEqual(board.writes, ['1000000000', '300000000'])
        for name in ('cur_freq', 'governor', 'min_freq', 'max_freq'):
            self.assertEqual(board.final[name], board.original[name])
        self.assertIs(report['userspace_request_was_undefined'], True)
        self.assertEqual(board.final['userspace/set_freq'], '300000000')
        self.assertFalse(board.group_alive)
        self.assertFalse(report['candidate_accepted'])
        self.assertEqual(report['restore_errors'], [])
        self.assertEqual(board.signal_calls[-3:], [(sig, signal.SIG_IGN)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)])

    def test_completion_restores_actual_frequency_and_records_undefined_difference(self):
        code, report, board = self.run_case('completed')
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'worker_completed_pending_result_review')
        self.assert_restored(report, board)

    def test_75c_stops_worker_and_restores(self):
        code, report, board = self.run_case('thermal')
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('Thermal stop threshold', report['error'])
        self.assertIn(signal.SIGTERM, board.kill_calls)
        self.assert_restored(report, board)

    def test_monitoring_read_failure_stops_worker_and_restores(self):
        code, report, board = self.run_case('read-error')
        self.assertEqual(code, 1)
        self.assertIn('Generated monitoring read failure', report['error'])
        self.assert_restored(report, board)

    def test_timeout_stops_worker_and_restores(self):
        code, report, board = self.run_case('timeout')
        self.assertEqual(code, 1)
        self.assertIn('Bounded test timeout', report['error'])
        self.assertLessEqual(board.now, 1.)
        self.assert_restored(report, board)

    def test_nonzero_launcher_does_not_report_completion(self):
        code, report, board = self.run_case('nonzero')
        self.assertEqual(code, 1)
        self.assertEqual(report['child_returncode'], 7)
        self.assertEqual(report['status'], 'failed')
        self.assert_restored(report, board)

    def test_exited_leader_with_live_descendant_is_killed_before_restore(self):
        code, report, board = self.run_case('leader-exited')
        self.assertEqual(code, 0)
        self.assertEqual(report['child_returncode'], 0)
        self.assertIn(signal.SIGKILL, board.kill_calls)
        self.assertGreaterEqual(board.now, 5.)
        self.assertLess(board.now, 5.2)
        self.assertLess(board.events.index(('signal', signal.SIGKILL)),
                        board.events.index(('frequency', '300000000')))
        self.assert_restored(report, board)

    def test_restore_failure_cannot_report_success(self):
        code, report, board = self.run_case('restore-error')
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'failed_restore')
        self.assertFalse(report.get('frequency_governor_limits_restored', False))
        self.assertIn('Generated restore write failure', report['restore_errors'][0])
        self.assertEqual(board.final['cur_freq'], '1000000000')

    def test_disappearing_known_thermal_sensor_must_fail_and_restore(self):
        code, report, board = self.run_case('missing-sensor')
        self.assertEqual(len(report['baseline_temperatures_c']), 7)
        self.assertEqual(code, 1, 'Losing a baseline thermal sensor must stop the test')
        self.assertEqual(report['status'], 'failed')
        self.assert_restored(report, board)


if __name__ == '__main__':
    unittest.main(verbosity=2)
