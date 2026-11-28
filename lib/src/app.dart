import 'package:material_ui/material_ui.dart';
import 'package:my_hero/src/core/routes/app_routes.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';

class GemmaChatApp extends StatelessWidget {
  const GemmaChatApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gemma Local Chat',
      theme: AppTheme.lightTheme,
      initialRoute: AppRoutes.initial,
      onGenerateRoute: AppRoutes.onGenerateRoute,
    );
  }
}
