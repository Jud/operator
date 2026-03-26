import AudioToolbox
import CoreAudio
import Foundation

/// Represents an audio device available on the system.
public struct AudioDevice: Identifiable, Hashable, Sendable {
    /// The unique CoreAudio device ID.
    public let id: AudioDeviceID

    /// The device UID string (stable across reboots).
    public let uid: String

    /// The human-readable device name.
    public let name: String
}

/// Enumerates system audio input and output devices via CoreAudio.
public enum AudioDeviceManager {
    /// Returns all available audio input devices.
    public static func inputDevices() -> [AudioDevice] {
        allDevices().filter { hasStreams(deviceID: $0.id, scope: kAudioObjectPropertyScopeInput) }
    }

    /// Look up an AudioDeviceID by its UID string.
    ///
    /// - Returns: The device ID, or nil if no device matches.
    public static func deviceID(forUID uid: String) -> AudioDeviceID? {
        allDevices().first { $0.uid == uid }?.id
    }

    /// Find the built-in microphone, preferring it over Bluetooth devices.
    ///
    /// Looks for devices with "Built-in" or "MacBook" in the name that have
    /// input streams. Returns nil if no built-in mic is found.
    public static func builtInMicID() -> AudioDeviceID? {
        let inputs = inputDevices()
        let builtIn = inputs.first {
            $0.name.contains("Built-in") || $0.name.contains("MacBook")
        }
        return builtIn?.id
    }

    // MARK: - Private

    private static func allDevices() -> [AudioDevice] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var size: UInt32 = 0
        guard
            AudioObjectGetPropertyDataSize(
                AudioObjectID(kAudioObjectSystemObject),
                &address,
                0,
                nil,
                &size
            ) == noErr
        else {
            return []
        }

        let count = Int(size) / MemoryLayout<AudioDeviceID>.size
        var deviceIDs = [AudioDeviceID](repeating: 0, count: count)

        guard
            AudioObjectGetPropertyData(
                AudioObjectID(kAudioObjectSystemObject),
                &address,
                0,
                nil,
                &size,
                &deviceIDs
            ) == noErr
        else {
            return []
        }

        return deviceIDs.compactMap { deviceID in
            guard let name = deviceName(deviceID: deviceID),
                let uid = deviceUID(deviceID: deviceID)
            else {
                return nil
            }
            return AudioDevice(id: deviceID, uid: uid, name: name)
        }
    }

    private static func deviceName(deviceID: AudioDeviceID) -> String? {
        stringProperty(deviceID: deviceID, selector: kAudioDevicePropertyDeviceNameCFString)
    }

    private static func deviceUID(deviceID: AudioDeviceID) -> String? {
        stringProperty(deviceID: deviceID, selector: kAudioDevicePropertyDeviceUID)
    }

    private static func stringProperty(
        deviceID: AudioDeviceID,
        selector: AudioObjectPropertySelector
    ) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        // Use a raw pointer to avoid UnsafeMutableRawPointer-to-Optional warnings.
        // CoreAudio writes a CFString reference into the buffer.
        var size = UInt32(MemoryLayout<CFTypeRef>.size)
        var ref: CFTypeRef?

        let status = withUnsafeMutablePointer(to: &ref) { ptr in
            AudioObjectGetPropertyData(
                deviceID,
                &address,
                0,
                nil,
                &size,
                ptr
            )
        }
        guard status == noErr, let value = ref as? String
        else {
            return nil
        }
        return value
    }

    private static func hasStreams(deviceID: AudioDeviceID, scope: AudioObjectPropertyScope) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreams,
            mScope: scope,
            mElement: kAudioObjectPropertyElementMain
        )

        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(deviceID, &address, 0, nil, &size) == noErr else {
            return false
        }
        return size > 0
    }
}
