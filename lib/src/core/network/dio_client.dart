import 'package:dio/dio.dart';

class DioClient {
  final Dio dio;

  DioClient([Dio? customDio])
      : dio = customDio ??
            Dio(
              BaseOptions(
                connectTimeout: const Duration(seconds: 30),
                receiveTimeout: const Duration(minutes: 10),
              ),
            );
}
