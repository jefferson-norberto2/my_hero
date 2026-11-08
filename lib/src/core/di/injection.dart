import 'package:get_it/get_it.dart';
import '../network/dio_client.dart';
import '../../shared/services/media_picker_service.dart';
import '../../features/chat/data/datasource/chat_history_datasource.dart';
import '../../features/chat/data/datasource/ollama_remote_datasource.dart';
import '../../features/chat/data/repository/chat_repository.dart';
import '../../features/chat/data/repository/chat_repository_impl.dart';
import '../../features/chat/controller/chat_controller.dart';

final getIt = GetIt.instance;

Future<void> setupDependencies() async {
  // Core & Shared
  getIt.registerLazySingleton<DioClient>(() => DioClient());
  getIt.registerLazySingleton<MediaPickerService>(() => MediaPickerService());

  // Features - Chat - Data
  getIt.registerLazySingleton<OllamaRemoteDatasource>(
    () => OllamaRemoteDatasourceImpl(getIt<DioClient>()),
  );
  getIt.registerLazySingleton<ChatHistoryDatasource>(
    () => ChatHistoryDatasourceImpl(),
  );
  getIt.registerLazySingleton<ChatRepository>(
    () => ChatRepositoryImpl(
      ollamaDatasource: getIt<OllamaRemoteDatasource>(),
      historyDatasource: getIt<ChatHistoryDatasource>(),
    ),
  );

  // Features - Chat - Controller
  getIt.registerFactory<ChatController>(
    () => ChatController(
      repository: getIt<ChatRepository>(),
      mediaPickerService: getIt<MediaPickerService>(),
    ),
  );
}
