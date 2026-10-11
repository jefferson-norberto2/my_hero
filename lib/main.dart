import 'package:flutter/material.dart';
import 'src/app.dart';
import 'src/core/di/injection.dart';
import 'src/features/chat/data/datasource/gemma_local_datasource.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await setupDependencies();
  await getIt<GemmaLocalDatasource>().initialize();
  runApp(const GemmaChatApp());
}
