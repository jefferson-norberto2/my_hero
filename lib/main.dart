import 'package:flutter/widgets.dart';
import 'package:my_hero/src/app.dart';
import 'package:my_hero/src/core/di/injection.dart';
import 'package:my_hero/src/features/chat/data/datasource/gemma_local_datasource.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await setupDependencies();
  await getIt<GemmaLocalDatasource>().initialize();
  runApp(const GemmaChatApp());
}
