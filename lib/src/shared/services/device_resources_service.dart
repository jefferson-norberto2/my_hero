import 'dart:async';
import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;
import 'package:flutter/services.dart';

/// Informações de memória RAM e armazenamento livre obtidas nativamente
/// no dispositivo (Android). Em outras plataformas ou ambiente de teste,
/// [DeviceResourcesService.getDeviceResources] pode retornar `null`.
class DeviceResourcesInfo {
  static const int _lowRamThresholdBytes = 6 * 1024 * 1024 * 1024;

  final int totalRamBytes;
  final int availableRamBytes;
  final bool lowMemory;
  final int memoryThresholdBytes;
  final int freeStorageBytes;

  const DeviceResourcesInfo({
    required this.totalRamBytes,
    required this.availableRamBytes,
    required this.lowMemory,
    required this.memoryThresholdBytes,
    required this.freeStorageBytes,
  });

  bool get hasRamInfo => totalRamBytes > 0;

  bool get hasStorageInfo => freeStorageBytes > 0;

  /// Considera o aparelho com RAM restrita quando possui até 6 GiB de RAM
  /// total ou o sistema Android sinaliza estado `lowMemory`.
  bool get isLowRamDevice =>
      hasRamInfo && (totalRamBytes <= _lowRamThresholdBytes || lowMemory);

  factory DeviceResourcesInfo.fromMap(Map<Object?, Object?> map) {
    return DeviceResourcesInfo(
      totalRamBytes: (map['totalRamBytes'] as num?)?.toInt() ?? 0,
      availableRamBytes: (map['availableRamBytes'] as num?)?.toInt() ?? 0,
      lowMemory: map['lowMemory'] as bool? ?? false,
      memoryThresholdBytes:
          (map['memoryThresholdBytes'] as num?)?.toInt() ?? 0,
      freeStorageBytes: (map['freeStorageBytes'] as num?)?.toInt() ?? 0,
    );
  }
}

enum ModelMemoryRisk { unknown, safe, warning, high }

class ModelMemoryAssessment {
  final ModelMemoryRisk risk;
  final int estimatedRequiredRamBytes;
  final String? message;

  const ModelMemoryAssessment({
    required this.risk,
    required this.estimatedRequiredRamBytes,
    this.message,
  });
}

abstract class DeviceResourcesService {
  Future<DeviceResourcesInfo?> getDeviceResources();
}

class NoopDeviceResourcesService implements DeviceResourcesService {
  const NoopDeviceResourcesService();

  @override
  Future<DeviceResourcesInfo?> getDeviceResources() async => null;
}

class PlatformDeviceResourcesService implements DeviceResourcesService {
  static const String channelName = 'com.example.my_hero/device_resources';

  final MethodChannel _channel;
  final TargetPlatform? _platformOverride;

  PlatformDeviceResourcesService({
    MethodChannel? channel,
    TargetPlatform? platform,
  })  : _channel = channel ?? const MethodChannel(channelName),
        _platformOverride = platform;

  bool get _isSupported {
    if (kIsWeb) return false;
    final effective = _platformOverride ?? defaultTargetPlatform;
    return effective == TargetPlatform.android;
  }

  @override
  Future<DeviceResourcesInfo?> getDeviceResources() async {
    if (!_isSupported) return null;
    try {
      final raw = await _channel.invokeMapMethod<Object?, Object?>(
        'getDeviceResources',
      );
      if (raw == null) return null;
      return DeviceResourcesInfo.fromMap(raw);
    } on MissingPluginException {
      return null;
    } on PlatformException {
      return null;
    }
  }
}
