import 'dart:async';
import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;
import 'package:flutter/services.dart';

/// Gerencia a notificação de Foreground Service durante a geração de respostas
/// no Android (mantendo o processo vivo enquanto o modelo pensa/responde em
/// segundo plano, com botão "Parar" e aviso de "Resposta pronta").
///
/// Em Linux e Windows todas as chamadas são no-op seguras.
abstract class InferenceNotificationService {
  void setOnStopRequested(Future<void> Function()? callback);

  Future<void> start({
    required String title,
    required String body,
  });

  Future<void> update({
    required String title,
    required String body,
  });

  Future<void> stop({
    bool notifyIfBackgrounded = false,
    String? completedTitle,
    String? completedBody,
  });

  Future<void> clearCompletedNotification();
}

class NoopInferenceNotificationService implements InferenceNotificationService {
  const NoopInferenceNotificationService();

  @override
  void setOnStopRequested(Future<void> Function()? callback) {}

  @override
  Future<void> start({required String title, required String body}) async {}

  @override
  Future<void> update({required String title, required String body}) async {}

  @override
  Future<void> stop({
    bool notifyIfBackgrounded = false,
    String? completedTitle,
    String? completedBody,
  }) async {}

  @override
  Future<void> clearCompletedNotification() async {}
}

class PlatformInferenceNotificationService
    implements InferenceNotificationService {
  static const String channelName = 'com.example.my_hero/inference_notification';

  final MethodChannel _channel;
  final TargetPlatform? _platformOverride;
  Future<void> Function()? _onStopRequested;

  PlatformInferenceNotificationService({
    MethodChannel? channel,
    TargetPlatform? platform,
  })  : _channel = channel ?? const MethodChannel(channelName),
        _platformOverride = platform {
    if (_isSupported) {
      _channel.setMethodCallHandler(_handleNativeCall);
    }
  }

  bool get _isSupported {
    if (kIsWeb) return false;
    final effective = _platformOverride ?? defaultTargetPlatform;
    return effective == TargetPlatform.android;
  }

  Future<void> _handleNativeCall(MethodCall call) async {
    if (call.method == 'onStopRequested') {
      final callback = _onStopRequested;
      if (callback != null) {
        await callback();
      }
    }
  }

  @override
  void setOnStopRequested(Future<void> Function()? callback) {
    _onStopRequested = callback;
  }

  @override
  Future<void> start({
    required String title,
    required String body,
  }) async {
    if (!_isSupported) return;
    await _invokeSafely('start', {
      'title': title,
      'body': body,
    });
  }

  @override
  Future<void> update({
    required String title,
    required String body,
  }) async {
    if (!_isSupported) return;
    await _invokeSafely('update', {
      'title': title,
      'body': body,
    });
  }

  @override
  Future<void> stop({
    bool notifyIfBackgrounded = false,
    String? completedTitle,
    String? completedBody,
  }) async {
    if (!_isSupported) return;
    await _invokeSafely('stop', {
      'notifyIfBackgrounded': notifyIfBackgrounded,
      'completedTitle': completedTitle,
      'completedBody': completedBody,
    });
  }

  @override
  Future<void> clearCompletedNotification() async {
    if (!_isSupported) return;
    await _invokeSafely('clearCompletedNotification');
  }

  Future<void> _invokeSafely(
    String method, [
    Map<String, Object?>? arguments,
  ]) async {
    try {
      await _channel.invokeMethod<void>(method, arguments);
    } on MissingPluginException {
      // Ambiente de teste ou plataforma sem canal nativo registrado.
    } on PlatformException {
      // Falha não-fatal na notificação nunca deve interromper a inferência.
    }
  }
}
