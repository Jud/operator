import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { existsSync, readFileSync } from "fs";
import { execSync } from "child_process";
import { homedir, platform } from "os";
import { basename } from "path";

const TOKEN = readFileSync(`${homedir()}/.operator/token`, "utf-8").trim();
const DAEMON_URL = "http://localhost:7420";

/**
 * Detect the TTY device of the parent process (Claude Code) by walking the
 * process tree via `ps`. Returns the device path and owning PID, or null.
 */
function detectParentTTY(): { device: string; pid: number } | null {
    const plat = platform();
    if (plat !== "darwin" && plat !== "linux") {
        return null;
    }

    try {
        let currentPid = String(process.ppid);

        for (let depth = 0; depth < 10; depth++) {
            const tty = execSync(`ps -o tty= -p ${currentPid}`, {
                encoding: "utf-8",
                timeout: 2000,
            }).trim();

            if (tty !== "" && tty !== "??" && tty !== "?") {
                const device = `/dev/${tty}`;
                if (existsSync(device)) {
                    return { device, pid: parseInt(currentPid, 10) };
                }
            }

            const ppidStr = execSync(`ps -o ppid= -p ${currentPid}`, {
                encoding: "utf-8",
                timeout: 2000,
            }).trim();

            currentPid = ppidStr;
            if (currentPid === "0" || currentPid === "1" || currentPid === "") {
                break;
            }
        }
    } catch {
        // ps command failed or process doesn't exist
    }

    return null;
}

// --- Module-level auto-detected state ---

let detectedTTY: string | undefined;
const detectedSessionName: string | undefined = basename(process.cwd());

const ttyResult = detectParentTTY();
if (ttyResult) {
    detectedTTY = ttyResult.device;
}

// --- Daemon helpers ---

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

async function daemonGet(path: string): Promise<unknown> {
    const res = await fetch(`${DAEMON_URL}${path}`, {
        headers: {
            // eslint-disable-next-line @typescript-eslint/naming-convention
            Authorization: `Bearer ${TOKEN}`,
        },
    });
    if (!res.ok) {
        throw new Error(`Daemon error: ${String(res.status)}`);
    }
    return res.json();
}

// --- Auto-registration ---

let registeredSessionName: string | undefined;

async function autoRegister(): Promise<void> {
    if (detectedTTY === undefined || detectedSessionName === undefined) {
        console.error("[operator-mcp] auto-register skipped: TTY or session name not detected");
        return;
    }

    try {
        await daemonPost("/register", {
            name: detectedSessionName,
            tty: detectedTTY,
            cwd: process.cwd(),
        });
        registeredSessionName = detectedSessionName;
        console.error(
            `[operator-mcp] auto-registered as "${detectedSessionName}" on ${detectedTTY}`,
        );
    } catch (err: unknown) {
        console.error(
            `[operator-mcp] auto-register failed (daemon may not be running): ${String(err)}`,
        );
    }
}

// --- Tools ---

const server = new McpServer({ name: "operator", version: "1.0.0" });

server.registerTool(
    "register",
    {
        description:
            "Register this Claude Code session with Operator. The TTY is auto-detected; only call this to override the session name or provide additional context.",
        inputSchema: {
            name: z
                .string()
                .describe(
                    "Human-readable session name (e.g. 'sudo', 'frontend'). Cannot be 'operator'.",
                ),
            tty: z
                .string()
                .optional()
                .describe("TTY path (auto-detected on startup; only provide to override)"),
            context: z.string().optional().describe("Brief description of current work"),
        },
    },
    async ({ name, tty, context }) => {
        if (name.toLowerCase() === "operator") {
            return {
                content: [{ type: "text" as const, text: "Error: 'operator' is reserved" }],
            };
        }
        const effectiveTTY = tty ?? detectedTTY;
        if (effectiveTTY === undefined) {
            return {
                content: [
                    {
                        type: "text" as const,
                        text: "Error: no TTY detected and none provided",
                    },
                ],
            };
        }
        registeredSessionName = name;
        const result = await daemonPost("/register", {
            name,
            tty: effectiveTTY,
            context,
            cwd: process.cwd(),
        });
        return {
            content: [{ type: "text" as const, text: JSON.stringify(result) }],
        };
    },
);

server.registerTool(
    "speak",
    {
        description:
            "Speak a message to the user through Operator's audio queue. Use this instead of the say command.",
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
            session: registeredSessionName,
        });
        return {
            content: [{ type: "text" as const, text: JSON.stringify(result) }],
        };
    },
);

server.registerTool(
    "update_context",
    {
        description:
            "Update session context for routing. Called after significant work. TTY is auto-detected; only provide to override.",
        inputSchema: {
            tty: z
                .string()
                .optional()
                .describe("TTY path for this session (auto-detected; only provide to override)"),
            summary: z.string().describe("Brief summary of current work"),
            recentMessages: z
                .array(
                    z.object({
                        role: z.string(),
                        text: z.string(),
                    }),
                )
                .optional()
                .describe("Recent conversation messages for routing context"),
        },
    },
    async ({ tty, summary, recentMessages }) => {
        const effectiveTTY = tty ?? detectedTTY;
        if (effectiveTTY === undefined) {
            return {
                content: [
                    {
                        type: "text" as const,
                        text: "Error: no TTY detected and none provided",
                    },
                ],
            };
        }
        const result = await daemonPost("/update", {
            tty: effectiveTTY,
            summary,
            // eslint-disable-next-line @typescript-eslint/naming-convention
            recent_messages: recentMessages,
        });
        return {
            content: [{ type: "text" as const, text: JSON.stringify(result) }],
        };
    },
);

// get_state is available via daemonGet("/state") but not exposed as a tool —
// the daemon handles coordination, individual sessions don't need to read state.
void daemonGet;

const transport = new StdioServerTransport();
await server.connect(transport);
await autoRegister();
