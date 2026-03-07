#!/usr/bin/env node
/**
 * Production-ready TTY detection for MCP servers spawned by Claude Code.
 *
 * DISPOSABLE - reference implementation for the actual MCP server.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const os = require('os');

/**
 * Detect the TTY device of the parent process (Claude Code).
 *
 * Strategy:
 * 1. Look up parent's TTY via `ps -o tty= -p <ppid>`
 * 2. If parent has no TTY (??), walk up the process tree
 * 3. Validate the device exists and is writable
 *
 * @returns {{ device: string, pid: number } | null}
 */
function detectParentTTY() {
  const platform = os.platform();

  if (platform !== 'darwin' && platform !== 'linux') {
    // Windows doesn't have /dev/ttyXXX
    return null;
  }

  try {
    let currentPid = process.ppid;

    for (let depth = 0; depth < 10; depth++) {
      const tty = execSync(`ps -o tty= -p ${currentPid}`, {
        encoding: 'utf-8',
        timeout: 2000,
      }).trim();

      if (tty && tty !== '??' && tty !== '?') {
        // On macOS: "ttys005" -> "/dev/ttys005"
        // On Linux: "pts/3" -> "/dev/pts/3"
        const device = `/dev/${tty}`;

        if (fs.existsSync(device)) {
          return { device, pid: parseInt(currentPid, 10) };
        }
      }

      // Walk up to parent's parent
      const ppidStr = execSync(`ps -o ppid= -p ${currentPid}`, {
        encoding: 'utf-8',
        timeout: 2000,
      }).trim();

      currentPid = ppidStr;
      if (currentPid === '0' || currentPid === '1' || !currentPid) {
        break;
      }
    }
  } catch {
    // ps command failed or process doesn't exist
  }

  return null;
}

/**
 * Write a message to the detected TTY.
 * This bypasses MCP's piped stdio and writes directly to the terminal.
 */
function writeToTTY(device, message) {
  try {
    const fd = fs.openSync(device, 'w');
    fs.writeSync(fd, message);
    fs.closeSync(fd);
    return true;
  } catch {
    return false;
  }
}

// --- Test ---
if (require.main === module) {
  const result = detectParentTTY();
  console.log('Detection result:', JSON.stringify(result, null, 2));

  if (result) {
    console.log(`\nSuccessfully detected TTY: ${result.device} (from PID ${result.pid})`);

    // Check permissions
    try {
      fs.accessSync(result.device, fs.constants.W_OK);
      console.log('TTY is writable: YES');
    } catch {
      console.log('TTY is writable: NO');
    }
  } else {
    console.log('\nFailed to detect parent TTY');
  }
}

module.exports = { detectParentTTY, writeToTTY };
