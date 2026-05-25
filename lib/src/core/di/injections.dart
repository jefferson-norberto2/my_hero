import 'package:dio/dio.dart';
import 'package:get_it/get_it.dart';
import 'package:my_hero/src/controller/chat_controller.dart';

final getIt = GetIt.instance;

void setupInjections() {
  getIt.registerSingleton<Dio>(Dio());
  getIt.registerSingleton<ChatController>(ChatController(getIt<Dio>()));
}