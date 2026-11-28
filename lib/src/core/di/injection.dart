import 'package:get_it/get_it.dart';
import 'package:my_hero/src/shared/services/inference_notification_service.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';
import 'package:my_hero/src/features/chat/data/datasource/chat_history_datasource.dart';
import 'package:my_hero/src/features/chat/data/datasource/gemma_local_datasource.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository_impl.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';

final getIt = GetIt.instance;

Future<void> setupDependencies() async {
  // Core & Shared
  getIt.registerLazySingleton<MediaPickerService>(() => MediaPickerService());
  getIt.registerLazySingleton<InferenceNotificationService>(
    () => PlatformInferenceNotificationService(),
  );

  // Features - Chat - Data
  getIt.registerLazySingleton<GemmaLocalDatasource>(
    () => GemmaLocalDatasourceImpl(),
  );
  getIt.registerLazySingleton<ChatHistoryDatasource>(
    () => ChatHistoryDatasourceImpl(),
  );
  getIt.registerLazySingleton<ChatRepository>(
    () => ChatRepositoryImpl(
      gemmaDatasource: getIt<GemmaLocalDatasource>(),
      historyDatasource: getIt<ChatHistoryDatasource>(),
    ),
  );

  // Features - Chat - Controller
  getIt.registerFactory<ChatController>(
    () => ChatController(
      repository: getIt<ChatRepository>(),
      mediaPickerService: getIt<MediaPickerService>(),
      inferenceNotificationService: getIt<InferenceNotificationService>(),
    ),
  );
}
