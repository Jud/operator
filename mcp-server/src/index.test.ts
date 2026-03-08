import { describe, it, expect, vi, beforeEach } from "vitest";
import { z } from "zod";
import { basename } from "path";

// The main module (index.ts) has top-level side effects (reads a token file,
// connects a stdio transport) that make direct import impractical in tests.
// Instead we test the extractable logic patterns: URL construction, auth header
// formatting, fetch behaviour, input validation, and response shaping.

const DAEMON_URL = "http://localhost:7420";

// Reproduce the speak tool's input schema exactly as declared in index.ts
const speakInputSchema = z.object({
    message: z.string().describe("Message to speak"),
    priority: z
        .enum(["normal", "urgent"])
        .default("normal")
        .describe("Priority level. Urgent messages skip to front of queue."),
});

interface MockResponse {
    ok: boolean;
    status?: number;
    json: () => Promise<unknown>;
}

describe("operator-mcp-server", () => {
    // ---------------------------------------------------------------
    // daemonPost logic
    // ---------------------------------------------------------------
    describe("daemonPost logic", () => {
        let mockFetch: ReturnType<typeof vi.fn<(...args: unknown[]) => Promise<MockResponse>>>;

        beforeEach(() => {
            mockFetch = vi.fn<(...args: unknown[]) => Promise<MockResponse>>();
        });

        // Mirrors the URL construction in daemonPost
        async function daemonPost(path: string, body: object, token: string): Promise<unknown> {
            const res = await mockFetch(`${DAEMON_URL}${path}`, {
                method: "POST",
                headers: {
                    // eslint-disable-next-line @typescript-eslint/naming-convention
                    "Content-Type": "application/json",
                    // eslint-disable-next-line @typescript-eslint/naming-convention
                    Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify(body),
            });
            if (!res.ok) {
                throw new Error(`Daemon error: ${String(res.status)}`);
            }
            return res.json();
        }

        it("constructs correct URL from path", async () => {
            mockFetch.mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({}),
            });

            await daemonPost("/speak", {}, "tok");

            expect(mockFetch).toHaveBeenCalledWith(
                "http://localhost:7420/speak",
                expect.anything(),
            );
        });

        it("constructs correct Authorization header", async () => {
            mockFetch.mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({}),
            });

            await daemonPost("/speak", {}, "test-token-123");

            expect(mockFetch).toHaveBeenCalledWith(
                expect.any(String) as string,
                expect.objectContaining({
                    headers: expect.objectContaining({
                        // eslint-disable-next-line @typescript-eslint/naming-convention
                        Authorization: "Bearer test-token-123",
                    }) as Record<string, string>,
                }),
            );
        });

        it("sends JSON body with correct Content-Type", async () => {
            mockFetch.mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ status: "queued" }),
            });

            const body = {
                message: "hello",
                priority: "normal",
                session: "myproject",
            };
            await daemonPost("/speak", body, "test-token");

            expect(mockFetch).toHaveBeenCalledWith(
                "http://localhost:7420/speak",
                expect.objectContaining({
                    method: "POST",
                    headers: expect.objectContaining({
                        // eslint-disable-next-line @typescript-eslint/naming-convention
                        "Content-Type": "application/json",
                    }) as Record<string, string>,
                    body: JSON.stringify(body),
                }),
            );
        });

        it("returns parsed JSON on success", async () => {
            mockFetch.mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ status: "queued", id: 42 }),
            });

            const result = await daemonPost("/speak", { message: "hi" }, "tok");
            expect(result).toEqual({ status: "queued", id: 42 });
        });

        it("throws on 401 response", async () => {
            mockFetch.mockResolvedValue({
                ok: false,
                status: 401,
                json: () => Promise.resolve({}),
            });

            await expect(daemonPost("/speak", { message: "hi" }, "bad-token")).rejects.toThrow(
                "Daemon error: 401",
            );
        });

        it("throws on 500 response", async () => {
            mockFetch.mockResolvedValue({
                ok: false,
                status: 500,
                json: () => Promise.resolve({}),
            });

            await expect(daemonPost("/speak", {}, "tok")).rejects.toThrow("Daemon error: 500");
        });

        it("throws on 404 response", async () => {
            mockFetch.mockResolvedValue({
                ok: false,
                status: 404,
                json: () => Promise.resolve({}),
            });

            await expect(daemonPost("/unknown", {}, "tok")).rejects.toThrow("Daemon error: 404");
        });

        it("propagates fetch network errors", async () => {
            mockFetch.mockRejectedValue(new TypeError("fetch failed"));

            await expect(daemonPost("/speak", {}, "tok")).rejects.toThrow("fetch failed");
        });
    });

    // ---------------------------------------------------------------
    // Session name derivation
    // ---------------------------------------------------------------
    describe("session name derivation", () => {
        it("derives session name from cwd basename", () => {
            expect(basename("/Users/jud/Projects/operator")).toBe("operator");
        });

        it("handles nested project paths", () => {
            expect(basename("/home/user/my-project")).toBe("my-project");
        });

        it("handles root path", () => {
            // basename("/") returns "" on POSIX
            expect(basename("/")).toBe("");
        });

        it("handles path with trailing slash", () => {
            expect(basename("/foo/bar/")).toBe("bar");
        });
    });

    // ---------------------------------------------------------------
    // Speak tool input schema validation
    // ---------------------------------------------------------------
    describe("speak tool input schema", () => {
        it("accepts message with default priority", () => {
            const result = speakInputSchema.safeParse({ message: "hello" });
            expect(result.success).toBe(true);
            if (result.success) {
                expect(result.data.message).toBe("hello");
                expect(result.data.priority).toBe("normal");
            }
        });

        it("rejects missing message", () => {
            const result = speakInputSchema.safeParse({});
            expect(result.success).toBe(false);
        });

        it("rejects non-string message", () => {
            const result = speakInputSchema.safeParse({ message: 123 });
            expect(result.success).toBe(false);
        });

        it("accepts urgent priority", () => {
            const result = speakInputSchema.safeParse({
                message: "alert!",
                priority: "urgent",
            });
            expect(result.success).toBe(true);
            if (result.success) {
                expect(result.data.priority).toBe("urgent");
            }
        });

        it("accepts normal priority explicitly", () => {
            const result = speakInputSchema.safeParse({
                message: "hi",
                priority: "normal",
            });
            expect(result.success).toBe(true);
            if (result.success) {
                expect(result.data.priority).toBe("normal");
            }
        });

        it("rejects invalid priority value", () => {
            const result = speakInputSchema.safeParse({
                message: "test",
                priority: "low",
            });
            expect(result.success).toBe(false);
        });

        it("rejects empty string as priority", () => {
            const result = speakInputSchema.safeParse({
                message: "test",
                priority: "",
            });
            expect(result.success).toBe(false);
        });
    });

    // ---------------------------------------------------------------
    // Speak tool response format
    // ---------------------------------------------------------------
    describe("speak tool response format", () => {
        // Mirrors the response construction in the speak tool handler
        function buildResponse(result: unknown): {
            content: { type: "text"; text: string }[];
        } {
            return {
                content: [{ type: "text" as const, text: JSON.stringify(result) }],
            };
        }

        it("wraps result as text content array", () => {
            const response = buildResponse({ status: "queued" });
            expect(response).toEqual({
                content: [{ type: "text", text: '{"status":"queued"}' }],
            });
        });

        it("serialises complex result objects", () => {
            const response = buildResponse({
                status: "queued",
                id: 7,
                eta: null,
            });
            const parsed: unknown = JSON.parse(response.content[0].text);
            expect(parsed).toEqual({ status: "queued", id: 7, eta: null });
        });

        it("content array has exactly one element", () => {
            const response = buildResponse({});
            expect(response.content).toHaveLength(1);
            expect(response.content[0].type).toBe("text");
        });
    });

    // ---------------------------------------------------------------
    // Token handling
    // ---------------------------------------------------------------
    describe("token handling", () => {
        it("trims leading and trailing whitespace", () => {
            const raw = "  my-secret-token  \n";
            expect(raw.trim()).toBe("my-secret-token");
        });

        it("trims newline-only padding", () => {
            const raw = "token-abc\n";
            expect(raw.trim()).toBe("token-abc");
        });

        it("leaves clean token unchanged", () => {
            const raw = "already-clean";
            expect(raw.trim()).toBe("already-clean");
        });
    });

    // ---------------------------------------------------------------
    // Speak tool body construction
    // ---------------------------------------------------------------
    describe("speak tool body construction", () => {
        it("includes message, priority, and session", () => {
            const sessionName = basename("/Users/jud/Projects/operator");
            const input = { message: "hello world", priority: "normal" as const };
            const body = {
                message: input.message,
                priority: input.priority,
                session: sessionName,
            };

            expect(body).toEqual({
                message: "hello world",
                priority: "normal",
                session: "operator",
            });
        });

        it("passes urgent priority through to body", () => {
            const input = speakInputSchema.parse({
                message: "fire!",
                priority: "urgent",
            });
            const body = { ...input, session: "myproject" };

            expect(body.priority).toBe("urgent");
            expect(body.session).toBe("myproject");
        });
    });
});
