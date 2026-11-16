import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';

/// Top-level entry point required by `flutter_foreground_task` on Android.
@pragma('vm:entry-point')
void startForegroundTaskCallback() {
  FlutterForegroundTask.setTaskHandler(_KeepAliveTaskHandler());
}

class _KeepAliveTaskHandler extends TaskHandler {
  @override
  Future<void> onStart(DateTime timestamp, TaskStarter starter) async {}

  @override
  void onRepeatEvent(DateTime timestamp) {}

  @override
  Future<void> onDestroy(DateTime timestamp, bool isTimeout) async {}
}

/// Platform abstraction over [FlutterForegroundTask] to enable unit testing
/// outside an Android runtime.
abstract class ForegroundTaskClient {
  void initCommunicationPort();

  void init({
    required AndroidNotificationOptions androidNotificationOptions,
    required IOSNotificationOptions iosNotificationOptions,
    required ForegroundTaskOptions foregroundTaskOptions,
  });

  Future<bool> get isRunningService;

  Future<NotificationPermission> checkNotificationPermission();

  Future<NotificationPermission> requestNotificationPermission();

  Future<ServiceRequestResult> startService({
    required int serviceId,
    required List<ForegroundServiceTypes> serviceTypes,
    required String notificationTitle,
    required String notificationText,
    required Function callback,
  });

  Future<ServiceRequestResult> updateService({
    required String notificationTitle,
    required String notificationText,
  });

  Future<ServiceRequestResult> stopService();
}

/// Default [ForegroundTaskClient] backed by [FlutterForegroundTask].
class FlutterForegroundTaskClient implements ForegroundTaskClient {
  const FlutterForegroundTaskClient();

  @override
  void initCommunicationPort() {
    FlutterForegroundTask.initCommunicationPort();
  }

  @override
  void init({
    required AndroidNotificationOptions androidNotificationOptions,
    required IOSNotificationOptions iosNotificationOptions,
    required ForegroundTaskOptions foregroundTaskOptions,
  }) {
    FlutterForegroundTask.init(
      androidNotificationOptions: androidNotificationOptions,
      iosNotificationOptions: iosNotificationOptions,
      foregroundTaskOptions: foregroundTaskOptions,
    );
  }

  @override
  Future<bool> get isRunningService => FlutterForegroundTask.isRunningService;

  @override
  Future<NotificationPermission> checkNotificationPermission() {
    return FlutterForegroundTask.checkNotificationPermission();
  }

  @override
  Future<NotificationPermission> requestNotificationPermission() {
    return FlutterForegroundTask.requestNotificationPermission();
  }

  @override
  Future<ServiceRequestResult> startService({
    required int serviceId,
    required List<ForegroundServiceTypes> serviceTypes,
    required String notificationTitle,
    required String notificationText,
    required Function callback,
  }) {
    return FlutterForegroundTask.startService(
      serviceId: serviceId,
      serviceTypes: serviceTypes,
      notificationTitle: notificationTitle,
      notificationText: notificationText,
      callback: callback,
    );
  }

  @override
  Future<ServiceRequestResult> updateService({
    required String notificationTitle,
    required String notificationText,
  }) {
    return FlutterForegroundTask.updateService(
      notificationTitle: notificationTitle,
      notificationText: notificationText,
    );
  }

  @override
  Future<ServiceRequestResult> stopService() {
    return FlutterForegroundTask.stopService();
  }
}

/// Manages the Android foreground service lifecycle during active chat streaming
/// and model download operations.
///
/// All platform calls are guarded by `Platform.isAndroid` so desktop targets
/// (such as Linux) remain completely unaffected.
class ForegroundTaskManager {
  /// Notification and foreground service ID on Android.
  static const int foregroundServiceId = 1001;

  static const String _notificationTitle = 'My Hero';
  static const String _channelId = 'my_hero_background_sync';
  static const String _channelName = 'Processamento em segundo plano';
  static const String _channelDescription =
      'Mantém a geração de respostas e downloads do Ollama ativos em segundo plano.';

  final ForegroundTaskClient _client;
  final bool Function() _isAndroid;

  bool _isInitialized = false;
  bool _hasRequestedNotificationPermission = false;
  bool _isChatActive = false;
  String? _activeDownloadModel;
  int? _activeDownloadPercent;
  String? _activeDownloadStatusText;
  String? _lastNotificationText;
  Future<void> _pendingOperation = Future<void>.value();

  ForegroundTaskManager({
    ForegroundTaskClient? client,
    bool Function()? isAndroid,
  })  : _client = client ?? const FlutterForegroundTaskClient(),
        _isAndroid = isAndroid ?? _defaultIsAndroid;

  static bool _defaultIsAndroid() => !kIsWeb && Platform.isAndroid;

  /// Initializes communication ports and notification channel configuration on Android.
  Future<void> initialize() {
    return _synchronized(() async {
      _ensureInitializedUnlocked();
    });
  }

  /// Marks chat generation as active and starts or updates the foreground service.
  Future<void> beginChatGeneration() {
    return _synchronized(() async {
      if (!_isAndroid()) return;
      _isChatActive = true;
      await _syncServiceStateUnlocked(requestPermissionIfNeeded: true);
    });
  }

  /// Marks chat generation as finished and stops the service if no download is active.
  Future<void> endChatGeneration() {
    return _synchronized(() async {
      if (!_isAndroid()) return;
      _isChatActive = false;
      await _syncServiceStateUnlocked(requestPermissionIfNeeded: false);
    });
  }

  /// Marks a model download as active and starts or updates the foreground service.
  Future<void> beginModelDownload(String modelName) {
    return _synchronized(() async {
      if (!_isAndroid()) return;
      _activeDownloadModel = modelName;
      _activeDownloadPercent = null;
      _activeDownloadStatusText = null;
      await _syncServiceStateUnlocked(requestPermissionIfNeeded: true);
    });
  }

  /// Updates the foreground notification with current download progress or status.
  Future<void> updateModelDownloadProgress(
    String modelName, {
    int? percent,
    String? statusText,
  }) {
    return _synchronized(() async {
      if (!_isAndroid() || _activeDownloadModel == null) return;
      _activeDownloadModel = modelName;
      _activeDownloadPercent = percent;
      _activeDownloadStatusText = statusText;
      await _syncServiceStateUnlocked(requestPermissionIfNeeded: false);
    });
  }

  /// Marks the active model download as finished and stops the service if idle.
  Future<void> endModelDownload() {
    return _synchronized(() async {
      if (!_isAndroid()) return;
      _activeDownloadModel = null;
      _activeDownloadPercent = null;
      _activeDownloadStatusText = null;
      await _syncServiceStateUnlocked(requestPermissionIfNeeded: false);
    });
  }

  Future<T> _synchronized<T>(Future<T> Function() action) {
    final completer = Completer<T>();
    _pendingOperation = _pendingOperation.then((_) async {
      try {
        completer.complete(await action());
      } catch (error, stackTrace) {
        completer.completeError(error, stackTrace);
      }
    });
    return completer.future;
  }

  void _ensureInitializedUnlocked() {
    if (!_isAndroid() || _isInitialized) return;
    try {
      _client.initCommunicationPort();
      _client.init(
        androidNotificationOptions: AndroidNotificationOptions(
          channelId: _channelId,
          channelName: _channelName,
          channelDescription: _channelDescription,
          channelImportance: NotificationChannelImportance.LOW,
          priority: NotificationPriority.LOW,
          onlyAlertOnce: true,
          showWhen: false,
          enableVibration: false,
          playSound: false,
        ),
        iosNotificationOptions: const IOSNotificationOptions(
          showNotification: false,
          playSound: false,
        ),
        foregroundTaskOptions: ForegroundTaskOptions(
          eventAction: ForegroundTaskEventAction.nothing(),
          autoRunOnBoot: false,
          autoRunOnMyPackageReplaced: false,
          allowWakeLock: true,
          allowWifiLock: true,
          allowAutoRestart: false,
          stopWithTask: true,
        ),
      );
      _isInitialized = true;
    } catch (_) {
      // Ignore initialization errors so network tasks still execute cleanly.
    }
  }

  Future<void> _requestNotificationPermissionIfNeededUnlocked() async {
    if (_hasRequestedNotificationPermission) return;
    _hasRequestedNotificationPermission = true;
    try {
      final status = await _client.checkNotificationPermission();
      if (status != NotificationPermission.granted) {
        await _client.requestNotificationPermission();
      }
    } catch (_) {
      // Starting a foreground service still works even if permission prompt is unavailable.
    }
  }

  Future<void> _syncServiceStateUnlocked({
    required bool requestPermissionIfNeeded,
  }) async {
    try {
      final hasActiveTask = _isChatActive || _activeDownloadModel != null;
      if (!hasActiveTask) {
        _lastNotificationText = null;
        if (await _client.isRunningService) {
          await _client.stopService();
        }
        return;
      }

      _ensureInitializedUnlocked();
      if (requestPermissionIfNeeded) {
        await _requestNotificationPermissionIfNeededUnlocked();
      }

      final notificationText = _buildNotificationText();
      final isRunning = await _client.isRunningService;

      if (!isRunning) {
        await _client.startService(
          serviceId: foregroundServiceId,
          serviceTypes: const [ForegroundServiceTypes.dataSync],
          notificationTitle: _notificationTitle,
          notificationText: notificationText,
          callback: startForegroundTaskCallback,
        );
        _lastNotificationText = notificationText;
      } else if (notificationText != _lastNotificationText) {
        await _client.updateService(
          notificationTitle: _notificationTitle,
          notificationText: notificationText,
        );
        _lastNotificationText = notificationText;
      }
    } catch (_) {
      // Do not fail data-layer operations if the OS rejects or interrupts the notification.
    }
  }

  String _buildNotificationText() {
    final modelName = _activeDownloadModel;
    String? downloadText;
    if (modelName != null) {
      final customStatus = _activeDownloadStatusText;
      if (customStatus != null && customStatus.isNotEmpty) {
        downloadText = customStatus;
      } else if (_activeDownloadPercent != null) {
        downloadText = 'Baixando modelo $modelName: $_activeDownloadPercent%';
      } else {
        downloadText = 'Baixando modelo $modelName...';
      }
    }

    if (downloadText != null && _isChatActive) {
      return '$downloadText • Gerando resposta...';
    }
    if (downloadText != null) {
      return downloadText;
    }
    return 'Gerando resposta...';
  }
}
