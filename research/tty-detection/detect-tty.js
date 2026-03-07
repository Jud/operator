#!/usr/bin/env node
/**
 * TTY Detection Experiment
 *
 * Goal: From inside a child process with piped stdio (like an MCP server),
 * detect the TTY of the parent process (Claude Code).
 *
 * DISPOSABLE - experimental code only.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const results = {};

// ========================================================================
// Approach 1: Use process.ppid to look up parent's TTY via `ps`
// ========================================================================
try {
  const ppid = process.ppid;
  const tty = execSync(`ps -o tty= -p ${ppid}`, { encoding: 'utf-8' }).trim();
  results['approach_1_ppid_ps'] = {
    method: 'process.ppid -> ps -o tty= -p <ppid>',
    ppid,
    tty: tty || '(empty)',
    device: tty ? `/dev/${tty}` : null,
    success: tty.length > 0 && tty !== '??',
  };
} catch (e) {
  results['approach_1_ppid_ps'] = { method: 'process.ppid -> ps', error: e.message, success: false };
}

// ========================================================================
// Approach 2: Walk the process tree upward until we find a TTY
// ========================================================================
try {
  let currentPid = process.ppid;
  const chain = [];
  let foundTty = null;

  for (let i = 0; i < 10; i++) {
    const info = execSync(`ps -o pid=,ppid=,tty=,comm= -p ${currentPid}`, { encoding: 'utf-8' }).trim();
    if (!info) break;

    const parts = info.split(/\s+/);
    const pid = parts[0];
    const ppid = parts[1];
    const tty = parts[2];
    const comm = parts.slice(3).join(' ');

    chain.push({ pid, ppid, tty, comm });

    if (tty && tty !== '??' && !foundTty) {
      foundTty = tty;
    }

    currentPid = ppid;
    if (currentPid === '0' || currentPid === '1') break;
  }

  results['approach_2_tree_walk'] = {
    method: 'Walk process tree upward, find first TTY',
    chain,
    foundTty,
    device: foundTty ? `/dev/${foundTty}` : null,
    success: foundTty !== null,
  };
} catch (e) {
  results['approach_2_tree_walk'] = { method: 'Tree walk', error: e.message, success: false };
}

// ========================================================================
// Approach 3: Check /dev/fd symlinks (likely fails for piped process)
// ========================================================================
try {
  const fdResults = {};
  for (const fd of [0, 1, 2]) {
    try {
      const target = fs.readlinkSync(`/dev/fd/${fd}`);
      fdResults[`fd${fd}`] = target;
    } catch (e) {
      fdResults[`fd${fd}`] = `error: ${e.message}`;
    }
  }
  results['approach_3_dev_fd'] = {
    method: 'readlinkSync(/dev/fd/N)',
    fds: fdResults,
    success: Object.values(fdResults).some(v => v.includes('/dev/tty')),
  };
} catch (e) {
  results['approach_3_dev_fd'] = { method: '/dev/fd', error: e.message, success: false };
}

// ========================================================================
// Approach 4: Check process.stdin.isTTY / process.stderr.isTTY
// ========================================================================
results['approach_4_istty'] = {
  method: 'process.std*.isTTY',
  stdin: process.stdin.isTTY || false,
  stdout: process.stdout.isTTY || false,
  stderr: process.stderr.isTTY || false,
  success: (process.stdin.isTTY || process.stdout.isTTY || process.stderr.isTTY) || false,
};

// ========================================================================
// Approach 5: Try opening /dev/tty directly (controlling terminal)
// ========================================================================
try {
  const fd = fs.openSync('/dev/tty', 'r');
  const stat = fs.fstatSync(fd);
  fs.closeSync(fd);
  results['approach_5_dev_tty'] = {
    method: 'Open /dev/tty directly',
    success: true,
    note: '/dev/tty is accessible',
  };
} catch (e) {
  results['approach_5_dev_tty'] = {
    method: 'Open /dev/tty directly',
    error: e.message,
    success: false,
    note: 'Process has no controlling terminal',
  };
}

// ========================================================================
// Approach 6: Find Claude process by name and get its TTY
// ========================================================================
try {
  const psOutput = execSync('ps -eo pid,tty,comm | grep -i claude | grep -v grep', { encoding: 'utf-8' }).trim();
  const lines = psOutput.split('\n').filter(Boolean);
  const claudeProcesses = lines.map(line => {
    const parts = line.trim().split(/\s+/);
    return { pid: parts[0], tty: parts[1], comm: parts.slice(2).join(' ') };
  });

  const withTty = claudeProcesses.filter(p => p.tty !== '??');

  results['approach_6_find_claude'] = {
    method: 'ps -eo pid,tty,comm | grep claude',
    claudeProcesses,
    processesWithTty: withTty,
    foundTty: withTty.length > 0 ? withTty[0].tty : null,
    device: withTty.length > 0 ? `/dev/${withTty[0].tty}` : null,
    success: withTty.length > 0,
    note: 'Fragile - depends on process name being "claude"',
  };
} catch (e) {
  results['approach_6_find_claude'] = { method: 'Find claude process', error: e.message, success: false };
}

// ========================================================================
// Approach 7: Environment variables that might contain TTY info
// ========================================================================
{
  const ttyVars = {};
  for (const [key, val] of Object.entries(process.env)) {
    if (/tty|term|pts|console/i.test(key) || /\/dev\/tty/i.test(val)) {
      ttyVars[key] = val;
    }
  }
  // Also check common ones explicitly
  ttyVars['TERM'] = process.env.TERM || '(unset)';
  ttyVars['TERM_PROGRAM'] = process.env.TERM_PROGRAM || '(unset)';
  ttyVars['SSH_TTY'] = process.env.SSH_TTY || '(unset)';
  ttyVars['GPG_TTY'] = process.env.GPP_TTY || '(unset)';

  results['approach_7_env_vars'] = {
    method: 'Check environment variables for TTY info',
    vars: ttyVars,
    success: Object.values(ttyVars).some(v => v.includes('/dev/tty')),
  };
}

// ========================================================================
// Approach 8: Check if /dev/ttyXXX from parent's ps output is readable
// ========================================================================
try {
  const ppid = process.ppid;
  const tty = execSync(`ps -o tty= -p ${ppid}`, { encoding: 'utf-8' }).trim();
  if (tty && tty !== '??') {
    const devPath = `/dev/${tty}`;
    const exists = fs.existsSync(devPath);
    let readable = false;
    let writable = false;
    try {
      fs.accessSync(devPath, fs.constants.R_OK);
      readable = true;
    } catch {}
    try {
      fs.accessSync(devPath, fs.constants.W_OK);
      writable = true;
    } catch {}

    results['approach_8_tty_access'] = {
      method: 'Check if parent TTY device is accessible',
      device: devPath,
      exists,
      readable,
      writable,
      success: exists && writable,
      note: writable ? 'Can write to parent TTY for notifications' : 'Cannot write to parent TTY',
    };
  }
} catch (e) {
  results['approach_8_tty_access'] = { method: 'TTY access check', error: e.message, success: false };
}

// ========================================================================
// Approach 9: Combined reliable method
// ========================================================================
try {
  // Best approach: ppid -> ps -> tty -> /dev/ttyXXX
  const ppid = process.ppid;
  const tty = execSync(`ps -o tty= -p ${ppid}`, { encoding: 'utf-8' }).trim();

  let detectedDevice = null;

  if (tty && tty !== '??') {
    detectedDevice = `/dev/${tty}`;
  } else {
    // Fallback: walk tree
    let currentPid = ppid;
    for (let i = 0; i < 10; i++) {
      const parentPpid = execSync(`ps -o ppid= -p ${currentPid}`, { encoding: 'utf-8' }).trim();
      const parentTty = execSync(`ps -o tty= -p ${parentPpid}`, { encoding: 'utf-8' }).trim();
      if (parentTty && parentTty !== '??') {
        detectedDevice = `/dev/${parentTty}`;
        break;
      }
      currentPid = parentPpid;
      if (currentPid === '0' || currentPid === '1') break;
    }
  }

  results['approach_9_combined'] = {
    method: 'Combined: ppid->ps with tree walk fallback',
    detectedDevice,
    success: detectedDevice !== null,
    note: 'Recommended production approach',
  };
} catch (e) {
  results['approach_9_combined'] = { method: 'Combined', error: e.message, success: false };
}

// ========================================================================
// Output
// ========================================================================
console.log(JSON.stringify(results, null, 2));

// Summary
console.log('\n=== SUMMARY ===');
for (const [key, val] of Object.entries(results)) {
  const status = val.success ? 'SUCCESS' : 'FAILED';
  const detail = val.device || val.detectedDevice || val.foundTty || val.error || '';
  console.log(`${status.padEnd(8)} ${key}: ${val.method} -> ${detail}`);
}
