import 'dart:io';
import 'package:path_provider/path_provider.dart';
import '../../../../core/constants/app_constants.dart';
import '../../../../core/network/dio_client.dart';

abstract class ModelDownloadDatasource {
  Future<String> getModelPath(String fileName);
  Future<bool> modelExists(String fileName);
  Future<String> downloadModel({
    required bool is2B,
    required void Function(double progress, String status) onProgress,
  });
}

class ModelDownloadDatasourceImpl implements ModelDownloadDatasource {
  final DioClient _dioClient;

  ModelDownloadDatasourceImpl(this._dioClient);

  @override
  Future<String> getModelPath(String fileName) async {
    final directory = await getApplicationDocumentsDirectory();
    return '${directory.path}/$fileName';
  }

  @override
  Future<bool> modelExists(String fileName) async {
    final path = await getModelPath(fileName);
    return File(path).exists();
  }

  @override
  Future<String> downloadModel({
    required bool is2B,
    required void Function(double progress, String status) onProgress,
  }) async {
    final fileName = is2B ? AppConstants.model2bFileName : AppConstants.model4bFileName;
    final downloadUrl = is2B ? AppConstants.model2bUrl : AppConstants.model4bUrl;
    final filePath = await getModelPath(fileName);
    final file = File(filePath);

    if (await file.exists()) {
      onProgress(1.0, 'Modelo $fileName já está disponível localmente.');
      return filePath;
    }

    onProgress(0.0, 'Iniciando download...');
    await _dioClient.dio.download(
      downloadUrl,
      filePath,
      onReceiveProgress: (received, total) {
        if (total != -1) {
          final progress = received / total;
          final receivedMb = (received / 1024 / 1024).toStringAsFixed(1);
          final totalMb = (total / 1024 / 1024).toStringAsFixed(1);
          onProgress(progress, 'Baixando: $receivedMb MB / $totalMb MB');
        }
      },
    );

    return filePath;
  }
}
