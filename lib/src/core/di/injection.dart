import 'package:get_it/get_it.dart';
import 'package:my_hero/src/core/network/dio_client.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/datasource/chat_history_datasource.dart';
import 'package:my_hero/src/features/chat/data/datasource/ollama_remote_datasource.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository_impl.dart';
import 'package:my_hero/src/features/settings/controller/settings_controller.dart';
import 'package:my_hero/src/features/settings/data/datasource/settings_datasource.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository_impl.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

final getIt = GetIt.instance;

/// Registers all core, shared, and feature dependencies in the service locator.
Future<void> setupDependencies() async {
  // Core & Shared
  getIt.registerLazySingleton<DioClient>(() => DioClient());
  getIt.registerLazySingleton<MediaPickerService>(() => MediaPickerService());

  // Features - Settings - Data
  getIt.registerLazySingleton<SettingsDatasource>(
    () => SettingsDatasourceImpl(),
  );

  // Features - Chat - Data
  getIt.registerLazySingleton<OllamaRemoteDatasource>(
    () => OllamaRemoteDatasourceImpl(getIt<DioClient>()),
  );
  getIt.registerLazySingleton<ChatHistoryDatasource>(
    () => ChatHistoryDatasourceImpl(),
  );
  getIt.registerLazySingleton<SettingsRepository>(
    () => SettingsRepositoryImpl(
      ollamaDatasource: getIt<OllamaRemoteDatasource>(),
      settingsDatasource: getIt<SettingsDatasource>(),
    ),
  );
  getIt.registerLazySingleton<ChatRepository>(
    () => ChatRepositoryImpl(
      ollamaDatasource: getIt<OllamaRemoteDatasource>(),
      historyDatasource: getIt<ChatHistoryDatasource>(),
      settingsDatasource: getIt<SettingsDatasource>(),
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
}
