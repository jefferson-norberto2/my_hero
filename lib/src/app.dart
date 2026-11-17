import 'package:flutter/material.dart';
import 'package:get_it/get_it.dart';
import 'package:go_router/go_router.dart';
import 'package:my_hero/src/core/routes/app_routes.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';
import 'package:my_hero/src/core/theme/theme_controller.dart';
import 'package:signals_flutter/signals_flutter.dart';

/// Root Material application configured with GoRouter and reactive theme state.
class GemmaChatApp extends StatelessWidget {
  final ThemeController? themeController;
  final GoRouter? routerConfig;

  const GemmaChatApp({
    super.key,
    this.themeController,
    this.routerConfig,
  });

  @override
  Widget build(BuildContext context) {
    final controller = themeController ??
        (GetIt.I.isRegistered<ThemeController>()
            ? GetIt.I<ThemeController>()
            : null);
    final router = routerConfig ?? AppRoutes.router;

    if (controller == null) {
      return MaterialApp.router(
        title: 'Gemma Local Chat',
        theme: AppTheme.lightTheme,
        darkTheme: AppTheme.darkTheme,
        themeMode: ThemeMode.light,
        routerConfig: router,
      );
    }

    return SignalBuilder(
      builder: (context) {
        final activeThemeMode = controller.themeMode.value;
        return MaterialApp.router(
          title: 'Gemma Local Chat',
          theme: AppTheme.lightTheme,
          darkTheme: AppTheme.darkTheme,
          themeMode: activeThemeMode,
          routerConfig: router,
        );
      },
    );
  }
}
