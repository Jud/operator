import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { readFileSync, writeFileSync } from "fs";
import { execSync, execFile } from "child_process";
import { promisify } from "util";
import { homedir } from "os";
import { basename } from "path";
import { randomUUID } from "crypto";

const TOKEN = readFileSync(`${homedir()}/.operator/token`, "utf-8").trim();
const DAEMON_URL = "http://localhost:7420";
const SESSION_NAME = basename(process.cwd());

// Detect terminal type from TERM_PROGRAM environment variable
const TERMINAL_TYPE: "ghostty" | "iterm2" =
    process.env.TERM_PROGRAM === "ghostty" ? "ghostty" : "iterm2";

// Detect TTY of the parent shell (same logic as the hook script)
function detectTTY(): string {
    try {
        const raw = execSync(`ps -o tty= -p ${String(process.ppid)}`, {
            encoding: "utf-8",
        }).trim();
        if (!raw) {
            return "unknown";
        }
        return raw.startsWith("/dev/") ? raw : `/dev/${raw}`;
    } catch {
        return "unknown";
    }
}

const TTY = detectTTY();

// Track whether Ghostty terminal ID resolution has been performed for this session
let terminalIdResolved = false;

// --- Daemon helpers ---

interface SessionStartResponse {
    ok: boolean;
    // eslint-disable-next-line @typescript-eslint/naming-convention
    needs_terminal_id?: boolean;
}

async function daemonPost(path: string, body: object): Promise<unknown> {
    const res = await fetch(`${DAEMON_URL}${path}`, {
        method: "POST",
        headers: {
            // eslint-disable-next-line @typescript-eslint/naming-convention
            "Content-Type": "application/json",
            // eslint-disable-next-line @typescript-eslint/naming-convention
            Authorization: `Bearer ${TOKEN}`,
        },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        throw new Error(`Daemon error: ${String(res.status)}`);
    }
    return res.json();
}

// --- Terminal ID Resolution ---

function delay(ms: number): Promise<void> {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}

async function resolveTerminalId(): Promise<void> {
    if (TTY === "unknown") {
        return;
    }
    const marker = `OPERATOR-${randomUUID()}`;
    try {
        // Write ANSI title escape directly to the terminal device file.
        // stdout is used by the MCP stdio transport, so we bypass it
        // and write to the TTY device to set the terminal title.
        writeFileSync(TTY, `\x1b]2;${marker}\x07`);

        // Allow Ghostty to process the title change
        await delay(500);

        const script = `(function() {
            var app = Application("Ghostty");
            var wins = app.windows();
            for (var i = 0; i < wins.length; i++) {
                var tabs = wins[i].tabs();
                for (var j = 0; j < tabs.length; j++) {
                    var terms = tabs[j].terminals();
                    for (var k = 0; k < terms.length; k++) {
                        if (terms[k].name().includes("${marker}")) {
                            return String(terms[k].id());
                        }
                    }
                }
            }
            return "not_found";
        })();`;

        const execFileAsync = promisify(execFile);
        const { stdout } = await execFileAsync("/usr/bin/osascript", [
            "-l",
            "JavaScript",
            "-e",
            script,
        ]);
        const ghosttyId = stdout.trim();

        if (ghosttyId !== "" && ghosttyId !== "not_found") {
            await daemonPost("/hook/terminal-id", {
                tty: TTY,
                // eslint-disable-next-line @typescript-eslint/naming-convention
                ghostty_id: ghosttyId,
            });
            terminalIdResolved = true;
        }
    } catch {
        // Resolution failed -- session remains TTY-keyed, delivery will fail gracefully
    }
}

// --- Tools ---

const server = new McpServer({ name: "operator", version: "1.0.0" });

server.registerTool(
    "speak",
    {
        description:
            "Send a short voice status update to the user. Call at end of every turn and at key milestones. One sentence max — like a walkie-talkie, not a presentation.",
        inputSchema: {
            message: z.string().describe("Message to speak"),
            priority: z
                .enum(["normal", "urgent"])
                .default("normal")
                .describe("Priority level. Urgent messages skip to front of queue."),
        },
    },
    async ({ message, priority }) => {
        const result = await daemonPost("/speak", {
            message,
            priority,
            session: SESSION_NAME,
        });
        return {
            content: [{ type: "text" as const, text: JSON.stringify(result) }],
        };
    },
);

// --- Heartbeat ---

// Re-register with the daemon every 5s so sessions survive Operator restarts.
async function heartbeat(): Promise<void> {
    try {
        const response = (await daemonPost("/hook/session-start", {
            // eslint-disable-next-line @typescript-eslint/naming-convention
            session_id: SESSION_NAME,
            tty: TTY,
            cwd: process.cwd(),
            // eslint-disable-next-line @typescript-eslint/naming-convention
            terminal_type: TERMINAL_TYPE,
        })) as SessionStartResponse;

        if (!terminalIdResolved && response.needs_terminal_id === true) {
            await resolveTerminalId();
        }
    } catch {
        // Daemon unavailable — will retry next interval.
    }
}

// Register immediately, then keep pinging.
// Greeting is handled by the Swift Heartbeat to avoid duplicates.
void heartbeat();
setInterval(() => void heartbeat(), 5_000);

const transport = new StdioServerTransport();
await server.connect(transport);
