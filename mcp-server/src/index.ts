import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { readFileSync } from "fs";
import { homedir } from "os";
import { basename } from "path";

const TOKEN = readFileSync(`${homedir()}/.operator/token`, "utf-8").trim();
const DAEMON_URL = "http://localhost:7420";
const SESSION_NAME = basename(process.cwd());

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

// --- Tools ---

const server = new McpServer({ name: "operator", version: "1.0.0" });

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
            session: SESSION_NAME,
        });
        return {
            content: [{ type: "text" as const, text: JSON.stringify(result) }],
        };
    },
);

const transport = new StdioServerTransport();
await server.connect(transport);
