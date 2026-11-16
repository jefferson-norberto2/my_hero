import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:get_it/get_it.dart';
import 'package:my_hero/src/core/network/dio_client.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/datasource/chat_history_datasource.dart';
import 'package:my_hero/src/features/chat/data/datasource/ollama_remote_datasource.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository_impl.dart';
import 'package:my_hero/src/features/chat/data/services/aes_gcm_encryption_service.dart';
import 'package:my_hero/src/features/chat/data/services/encryption_key_manager.dart';
import 'package:my_hero/src/features/chat/data/services/foreground_task_manager.dart';
import 'package:my_hero/src/features/settings/controller/settings_controller.dart';
import 'package:my_hero/src/features/settings/data/datasource/settings_datasource.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository_impl.dart';
import 'package:my_hero/src/features/settings/data/services/model_download_service.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

final getIt = GetIt.instance;

/// Registers all core, shared, and feature dependencies in the service locator.
Future<void> setupDependencies() async {
  // Core & Shared
  getIt.registerLazySingleton<DioClient>(() => DioClient());
  getIt.registerLazySingleton<MediaPickerService>(() => MediaPickerService());
  getIt.registerLazySingleton<FlutterSecureStorage>(
    () => const FlutterSecureStorage(
      aOptions: AndroidOptions(),
      lOptions: LinuxOptions(),
    ),
  );

  // Features - Settings - Data
  getIt.registerLazySingleton<SettingsDatasource>(
    () => SettingsDatasourceImpl(),
  );

  // Features - Chat - Data
  getIt.registerLazySingleton<ForegroundTaskManager>(
    () => ForegroundTaskManager(),
  );
  getIt.registerLazySingleton<EncryptionKeyManager>(
    () => EncryptionKeyManager(secureStorage: getIt<FlutterSecureStorage>()),
  );
  getIt.registerLazySingleton<AesGcmEncryptionService>(
    () => AesGcmEncryptionService(keyManager: getIt<EncryptionKeyManager>()),
  );
  getIt.registerLazySingleton<OllamaRemoteDatasource>(
    () => OllamaRemoteDatasourceImpl(getIt<DioClient>()),
  );
  getIt.registerLazySingleton<ChatHistoryDatasource>(
    () => ChatHistoryDatasourceImpl(
      encryptionService: getIt<AesGcmEncryptionService>(),
    ),
  );
  getIt.registerLazySingleton<ModelDownloadService>(
    () => ModelDownloadService(
      ollamaDatasource: getIt<OllamaRemoteDatasource>(),
      foregroundTaskManager: getIt<ForegroundTaskManager>(),
    ),
  );
  getIt.registerLazySingleton<SettingsRepository>(
    () => SettingsRepositoryImpl(
      ollamaDatasource: getIt<OllamaRemoteDatasource>(),
      settingsDatasource: getIt<SettingsDatasource>(),
      modelDownloadService: getIt<ModelDownloadService>(),
    ),
  );
  getIt.registerLazySingleton<ChatRepository>(
    () => ChatRepositoryImpl(
      ollamaDatasource: getIt<OllamaRemoteDatasource>(),
      historyDatasource: getIt<ChatHistoryDatasource>(),
      settingsDatasource: getIt<SettingsDatasource>(),
      foregroundTaskManager: getIt<ForegroundTaskManager>(),
    ),
  );

  // Features - Settings - Controller
  getIt.registerFactory<SettingsController>(
    () => SettingsController(repository: getIt<SettingsRepository>()),
  );

  // Features - Chat - Controller
  getIt.registerFactory<ChatController>(
    () => ChatController(
      repository: getIt<ChatRepository>(),
      mediaPickerService: getIt<MediaPickerService>(),
    ),
  );

  // Initialize Android foreground task configuration (no-op on desktop platforms).
  await getIt<ForegroundTaskManager>().initialize();

  // Initialize encrypted conversation storage and migrate any legacy JSON files.
  try {
    await getIt<ChatRepository>().initialize();
  } catch (_) {
    // Allow the UI to boot even if the platform keyring is temporarily unavailable.
  }
}
