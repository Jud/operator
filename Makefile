APP_NAME := Operator
BUILD_DIR := build
APP_BUNDLE := $(BUILD_DIR)/$(APP_NAME).app
CONTENTS := $(APP_BUNDLE)/Contents
MACOS := $(CONTENTS)/MacOS
RESOURCES := $(CONTENTS)/Resources
SWIFT_BUILD := Operator/.build/release
VERSION := $(shell git describe --tags --always 2>/dev/null || echo "0.1.0")

.PHONY: all clean app zip verify bench bench-list bench-routing bench-routing-latency bench-routing-accuracy restart

all: app

# Build the complete .app bundle
app: swift-build
	@echo "==> Assembling $(APP_NAME).app"
	mkdir -p $(MACOS) $(RESOURCES)

	# Daemon binary
	cp $(SWIFT_BUILD)/Operator $(MACOS)/Operator

	# MCP server binary
	cp $(SWIFT_BUILD)/OperatorMCP $(MACOS)/operator-mcp

	# SPM resource bundle (audio cues).
	# SPM's generated Bundle.module looks at Bundle.main.bundleURL/ (the .app root).
	cp -R $(SWIFT_BUILD)/Operator_OperatorCore.bundle $(APP_BUNDLE)/

	# Info.plist & PkgInfo
	cp Operator/Info.plist $(CONTENTS)/Info.plist
	echo -n "APPL????" > $(CONTENTS)/PkgInfo

	# Entitlements (for reference; used at signing time)
	cp Operator/Operator.entitlements $(CONTENTS)/

	# Codesign binaries outside the .app then copy back — codesign refuses
	# to sign inside an .app with unsealed root contents (the SPM resource
	# bundle at the .app root).

	# Main daemon binary: signed with stable identity so macOS preserves TCC
	# permissions (mic, accessibility) across rebuilds.
	cp $(MACOS)/Operator $(MACOS)/Operator.signing
	codesign --force --sign "Apple Development: Jud Stephenson (3SZVREW2PR)" \
		--entitlements Operator/Operator.entitlements \
		$(MACOS)/Operator.signing
	mv $(MACOS)/Operator.signing $(MACOS)/Operator

	# MCP server binary: ad-hoc signed to satisfy macOS code signature
	# validation (unsigned binaries get SIGKILL).
	cp $(MACOS)/operator-mcp $(MACOS)/operator-mcp.signing
	codesign --force --sign - $(MACOS)/operator-mcp.signing
	mv $(MACOS)/operator-mcp.signing $(MACOS)/operator-mcp

	@echo "==> $(APP_BUNDLE) assembled (signed)"

# Build the Swift daemon and MCP server
swift-build:
	@echo "==> Building Swift targets (release)"
	cd Operator && swift build -c release --disable-sandbox

# Build, kill, and restart the app
restart:
	./scripts/restart.sh

# Create distributable zip
zip: app
	@echo "==> Creating Operator-$(VERSION).zip"
	cd $(BUILD_DIR) && zip -r -y Operator-$(VERSION).zip $(APP_NAME).app
	@echo "==> $(BUILD_DIR)/Operator-$(VERSION).zip"

clean:
	rm -rf $(BUILD_DIR)
	cd Operator && swift package clean

verify:
	./scripts/verify-all.sh

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
