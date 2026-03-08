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

    /// Returns all available audio output devices.
    public static func outputDevices() -> [AudioDevice] {
        allDevices().filter { hasStreams(deviceID: $0.id, scope: kAudioObjectPropertyScopeOutput) }
    }

    /// Look up an AudioDeviceID by its UID string.
    ///
    /// - Returns: The device ID, or nil if no device matches.
    public static func deviceID(forUID uid: String) -> AudioDeviceID? {
        allDevices().first { $0.uid == uid }?.id
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

        var result: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)

        guard AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &result) == noErr,
            let cfString = result
        else {
            return nil
        }
        return cfString.takeUnretainedValue() as String
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
