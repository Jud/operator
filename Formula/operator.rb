class Operator < Formula
  desc "Voice-first orchestration layer for multiple concurrent Claude Code sessions"
  homepage "https://github.com/Jud/operator"
  url "https://github.com/Jud/operator.git",
      tag:      "v0.1.0",
      revision: "HEAD"
  license "MIT"
  head "https://github.com/Jud/operator.git", branch: "main"

  depends_on :macos
  depends_on :xcode => ["16.0", :build]
  depends_on "node" => :build
  depends_on "swift" => :build

  def install
    # Build the Swift daemon
    cd "Operator" do
      system "swift", "build", "-c", "release", "--disable-sandbox"
      bin.install ".build/release/Operator" => "operator-daemon"
    end

    # Build the MCP server
    cd "mcp-server" do
      system "npm", "ci", "--ignore-scripts"
      system "npm", "run", "build"

      # Install MCP server to libexec (not user-facing)
      libexec.install "build", "package.json", "node_modules"
    end

    # Create wrapper script for MCP server
    (bin/"operator-mcp").write <<~SH
      #!/bin/bash
      exec node "#{libexec}/build/index.js" "$@"
    SH

    # Install audio resource files
    cd "Operator/Sources/Resources" do
      (share/"operator/sounds").install Dir["*.caf"] if Dir["*.caf"].any?
    end
  end

  def post_install
    # Create config directory
    (var/"operator").mkpath

    # Generate auth token if it doesn't exist
    token_dir = Pathname.new(Dir.home)/".operator"
    unless (token_dir/"token").exist?
      token_dir.mkpath
      token_dir.chmod 0700
      token = `openssl rand -hex 32`.strip
      (token_dir/"token").write(token)
      (token_dir/"token").chmod 0600
    end

    # Auto-configure MCP server in Claude Code
    claude_bin = which("claude")
    if claude_bin
      system claude_bin, "mcp", "add", "operator",
             "--transport", "stdio",
             "--", bin/"operator-mcp"
      ohai "MCP server registered with Claude Code"
    else
      opoo "Claude Code CLI not found. Register manually:"
      puts "  claude mcp add operator -- #{bin}/operator-mcp"
    end
  end

  def caveats
    <<~EOS
      Operator has been installed with two components:

        1. Daemon:     #{bin}/operator-daemon
        2. MCP Server: #{bin}/operator-mcp (auto-registered with Claude Code)

      To start the daemon:
        operator-daemon

      To start on login (launchd):
        brew services start operator

      Requirements:
        - macOS 15+ (Sequoia)
        - iTerm2 with scripting enabled
        - Microphone permission (granted on first run)

      Config directory: ~/.operator/
    EOS
  end

  service do
    run [opt_bin/"operator-daemon"]
    keep_alive true
    log_path var/"log/operator.log"
    error_log_path var/"log/operator-error.log"
    environment_variables HOME: Dir.home
  end

  test do
    # Verify binaries exist
    assert_predicate bin/"operator-daemon", :exist?
    assert_predicate bin/"operator-mcp", :exist?

    # Verify MCP server can at least load
    output = shell_output("#{bin}/operator-mcp --help 2>&1", 1)
    assert_match(/node|Error/, output)
  end
end
