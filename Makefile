APP_NAME := Operator
BUILD_DIR := build
APP_BUNDLE := $(BUILD_DIR)/$(APP_NAME).app
CONTENTS := $(APP_BUNDLE)/Contents
MACOS := $(CONTENTS)/MacOS
RESOURCES := $(CONTENTS)/Resources
SWIFT_BUILD := Operator/.build/release
VERSION := $(shell git describe --tags --always 2>/dev/null || echo "0.1.0")
NODE := $(shell nodenv which node 2>/dev/null || command -v node 2>/dev/null || echo "node")

.PHONY: all clean app mcp zip bench bench-list bench-routing bench-routing-latency bench-routing-accuracy bench-tts bench-stt bench-stt-latency bench-stt-long bench-memory

all: app

# Build the complete .app bundle
app: swift-build mcp-build
	@echo "==> Assembling $(APP_NAME).app"
	mkdir -p $(MACOS) $(RESOURCES)

	# Binary
	cp $(SWIFT_BUILD)/Operator $(MACOS)/Operator

	# SPM resource bundle (audio cues)
	# Must be at .app root — SPM's Bundle.module looks at Bundle.main.bundleURL/
	cp -R $(SWIFT_BUILD)/Operator_OperatorCore.bundle $(APP_BUNDLE)/

	# Info.plist & PkgInfo
	cp Operator/Info.plist $(CONTENTS)/Info.plist
	echo -n "APPL????" > $(CONTENTS)/PkgInfo

	# Entitlements (for reference; used at signing time)
	cp Operator/Operator.entitlements $(CONTENTS)/

	# MCP server
	mkdir -p $(RESOURCES)/mcp-server
	cp -R mcp-server/build $(RESOURCES)/mcp-server/
	cp mcp-server/package.json $(RESOURCES)/mcp-server/
	cp -R mcp-server/node_modules $(RESOURCES)/mcp-server/

	# CLI wrapper: operator-mcp (resolves symlinks before finding Resources)
	@echo '#!/bin/bash' > $(MACOS)/operator-mcp
	@echo 'SELF="$$0"' >> $(MACOS)/operator-mcp
	@echo 'while [ -L "$$SELF" ]; do SELF="$$(readlink "$$SELF")"; done' >> $(MACOS)/operator-mcp
	@echo 'RESOURCES="$$(cd "$$(dirname "$$SELF")/../Resources" && pwd)"' >> $(MACOS)/operator-mcp
	@echo 'exec "$(NODE)" "$$RESOURCES/mcp-server/build/index.js" "$$@"' >> $(MACOS)/operator-mcp
	chmod +x $(MACOS)/operator-mcp

	@echo "==> $(APP_BUNDLE) assembled"

# Build the Swift daemon
swift-build:
	@echo "==> Building Swift daemon (release)"
	cd Operator && swift build -c release --disable-sandbox

# Build the MCP server
mcp-build:
	@echo "==> Building MCP server"
	cd mcp-server && npm ci --ignore-scripts && npm run build

# Create distributable zip
zip: app
	@echo "==> Creating Operator-$(VERSION).zip"
	cd $(BUILD_DIR) && zip -r -y Operator-$(VERSION).zip $(APP_NAME).app
	@echo "==> $(BUILD_DIR)/Operator-$(VERSION).zip"

clean:
	rm -rf $(BUILD_DIR)
	cd Operator && swift package clean

bench:
	./scripts/run-benchmarks.sh $(BENCH_ARGS) $(TARGET)

bench-list:
	./scripts/run-benchmarks.sh list

bench-routing:
	./scripts/run-benchmarks.sh routing

bench-routing-latency:
	./scripts/run-benchmarks.sh routing-latency

bench-routing-accuracy:
	./scripts/run-benchmarks.sh routing-accuracy

bench-tts:
	./scripts/run-benchmarks.sh tts

bench-stt:
	./scripts/run-benchmarks.sh stt

bench-stt-latency:
	./scripts/run-benchmarks.sh stt-latency

bench-stt-long:
	./scripts/run-benchmarks.sh stt-long

bench-memory:
	./scripts/run-benchmarks.sh memory
