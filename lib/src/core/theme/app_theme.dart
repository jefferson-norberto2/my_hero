import 'package:flutter/material.dart';

/// Semantic color tokens registered as a [ThemeExtension] so feature widgets
/// adapt cleanly between the default light theme and the Dracula dark theme.
@immutable
class AppThemeColors extends ThemeExtension<AppThemeColors> {
  final Color accent;
  final Color linkColor;
  final Color success;
  final Color warning;
  final Color danger;
  final Color infoSurface;
  final Color infoBorder;
  final Color infoIcon;
  final Color infoText;
  final Color infoSubtext;
  final Color audioAccent;
  final Color audioSurface;
  final Color documentAccent;
  final Color documentSurface;
  final Color userBubbleBackground;
  final Color userBubbleBorder;
  final Color assistantBubbleBackground;
  final Color assistantBubbleBorder;
  final Color sectionCardBackground;
  final Color sectionCardBorder;
  final Color chipSelectedBackground;
  final Color chipSelectedText;
  final Color chipSelectedBorder;
  final Color badgeSurface;
  final Color contextLow;
  final Color contextMedium;
  final Color contextHigh;
  final Color gpuTagBackground;
  final Color gpuTagForeground;
  final Color gpuTagBorder;
  final Color cpuTagBackground;
  final Color cpuTagForeground;
  final Color cpuTagBorder;
  final Color hybridTagBackground;
  final Color hybridTagForeground;
  final Color hybridTagBorder;
  final Color neutralTagBackground;
  final Color neutralTagForeground;
  final Color neutralTagBorder;

  const AppThemeColors({
    required this.accent,
    required this.linkColor,
    required this.success,
    required this.warning,
    required this.danger,
    required this.infoSurface,
    required this.infoBorder,
    required this.infoIcon,
    required this.infoText,
    required this.infoSubtext,
    required this.audioAccent,
    required this.audioSurface,
    required this.documentAccent,
    required this.documentSurface,
    required this.userBubbleBackground,
    required this.userBubbleBorder,
    required this.assistantBubbleBackground,
    required this.assistantBubbleBorder,
    required this.sectionCardBackground,
    required this.sectionCardBorder,
    required this.chipSelectedBackground,
    required this.chipSelectedText,
    required this.chipSelectedBorder,
    required this.badgeSurface,
    required this.contextLow,
    required this.contextMedium,
    required this.contextHigh,
    required this.gpuTagBackground,
    required this.gpuTagForeground,
    required this.gpuTagBorder,
    required this.cpuTagBackground,
    required this.cpuTagForeground,
    required this.cpuTagBorder,
    required this.hybridTagBackground,
    required this.hybridTagForeground,
    required this.hybridTagBorder,
    required this.neutralTagBackground,
    required this.neutralTagForeground,
    required this.neutralTagBorder,
  });

  static final ColorScheme _lightSeedScheme =
      ColorScheme.fromSeed(seedColor: Colors.teal);

  /// Preserves the original light-theme palette across all custom widgets.
  static final AppThemeColors light = AppThemeColors(
    accent: Colors.teal,
    linkColor: Colors.teal.shade700,
    success: Colors.green,
    warning: Colors.amber.shade800,
    danger: Colors.red,
    infoSurface: Colors.teal.shade50,
    infoBorder: Colors.teal.shade200,
    infoIcon: Colors.teal.shade700,
    infoText: Colors.teal.shade900,
    infoSubtext: Colors.teal.shade800,
    audioAccent: Colors.amber.shade800,
    audioSurface: Colors.amber.shade100,
    documentAccent: Colors.blue,
    documentSurface: Colors.blue.shade100,
    userBubbleBackground:
        _lightSeedScheme.primaryContainer.withValues(alpha: 0.6),
    userBubbleBorder: _lightSeedScheme.primary.withValues(alpha: 0.15),
    assistantBubbleBackground:
        _lightSeedScheme.surfaceContainerHighest.withValues(alpha: 0.5),
    assistantBubbleBorder:
        _lightSeedScheme.outlineVariant.withValues(alpha: 0.4),
    sectionCardBackground:
        _lightSeedScheme.surfaceContainerHighest.withValues(alpha: 0.4),
    sectionCardBorder: _lightSeedScheme.outlineVariant.withValues(alpha: 0.5),
    chipSelectedBackground: Colors.teal.withValues(alpha: 0.18),
    chipSelectedText: Colors.teal.shade900,
    chipSelectedBorder: Colors.teal,
    badgeSurface:
        _lightSeedScheme.surfaceContainerHighest.withValues(alpha: 0.8),
    contextLow: Colors.teal,
    contextMedium: Colors.amber.shade700,
    contextHigh: Colors.red.shade400,
    gpuTagBackground: const Color(0xFFD8F3DC),
    gpuTagForeground: const Color(0xFF1B4332),
    gpuTagBorder: const Color(0xFF95D5B2),
    cpuTagBackground: const Color(0xFFE2E8F0),
    cpuTagForeground: const Color(0xFF334155),
    cpuTagBorder: const Color(0xFFCBD5E1),
    hybridTagBackground: const Color(0xFFFEF3C7),
    hybridTagForeground: const Color(0xFF78350F),
    hybridTagBorder: const Color(0xFFFCD34D),
    neutralTagBackground: const Color(0xFFE5E7EB),
    neutralTagForeground: const Color(0xFF374151),
    neutralTagBorder: const Color(0xFFD1D5DB),
  );

  /// Dracula-inspired dark palette tailored to the app's visual hierarchy.
  static final AppThemeColors dracula = AppThemeColors(
    accent: const Color(0xFFBD93F9),
    linkColor: const Color(0xFF8BE9FD),
    success: const Color(0xFF50FA7B),
    warning: const Color(0xFFFFB86C),
    danger: const Color(0xFFFF5555),
    infoSurface: const Color(0xFF44475A).withValues(alpha: 0.55),
    infoBorder: const Color(0xFF6272A4).withValues(alpha: 0.5),
    infoIcon: const Color(0xFF8BE9FD),
    infoText: const Color(0xFFF8F8F2),
    infoSubtext: const Color(0xFFB6C2E2),
    audioAccent: const Color(0xFFFFB86C),
    audioSurface: const Color(0xFF44475A),
    documentAccent: const Color(0xFF8BE9FD),
    documentSurface: const Color(0xFF44475A),
    userBubbleBackground: const Color(0xFF44475A),
    userBubbleBorder: const Color(0xFFBD93F9).withValues(alpha: 0.35),
    assistantBubbleBackground: const Color(0xFF343746),
    assistantBubbleBorder: const Color(0xFF6272A4).withValues(alpha: 0.45),
    sectionCardBackground: const Color(0xFF343746),
    sectionCardBorder: const Color(0xFF6272A4).withValues(alpha: 0.4),
    chipSelectedBackground: const Color(0xFFBD93F9).withValues(alpha: 0.22),
    chipSelectedText: const Color(0xFFF8F8F2),
    chipSelectedBorder: const Color(0xFFBD93F9),
    badgeSurface: const Color(0xFF343746),
    contextLow: const Color(0xFF50FA7B),
    contextMedium: const Color(0xFFFFB86C),
    contextHigh: const Color(0xFFFF5555),
    gpuTagBackground: const Color(0xFF50FA7B).withValues(alpha: 0.16),
    gpuTagForeground: const Color(0xFF50FA7B),
    gpuTagBorder: const Color(0xFF50FA7B).withValues(alpha: 0.45),
    cpuTagBackground: const Color(0xFF8BE9FD).withValues(alpha: 0.15),
    cpuTagForeground: const Color(0xFF8BE9FD),
    cpuTagBorder: const Color(0xFF8BE9FD).withValues(alpha: 0.45),
    hybridTagBackground: const Color(0xFFFFB86C).withValues(alpha: 0.16),
    hybridTagForeground: const Color(0xFFFFB86C),
    hybridTagBorder: const Color(0xFFFFB86C).withValues(alpha: 0.45),
    neutralTagBackground: const Color(0xFF44475A),
    neutralTagForeground: const Color(0xFFB6C2E2),
    neutralTagBorder: const Color(0xFF6272A4).withValues(alpha: 0.5),
  );

  /// Safely resolves [AppThemeColors] from [context], falling back based on
  /// theme brightness for isolated widget tests that omit [AppTheme].
  static AppThemeColors of(BuildContext context) {
    final theme = Theme.of(context);
    return theme.extension<AppThemeColors>() ??
        (theme.brightness == Brightness.dark ? dracula : light);
  }

  @override
  AppThemeColors copyWith({
    Color? accent,
    Color? linkColor,
    Color? success,
    Color? warning,
    Color? danger,
    Color? infoSurface,
    Color? infoBorder,
    Color? infoIcon,
    Color? infoText,
    Color? infoSubtext,
    Color? audioAccent,
    Color? audioSurface,
    Color? documentAccent,
    Color? documentSurface,
    Color? userBubbleBackground,
    Color? userBubbleBorder,
    Color? assistantBubbleBackground,
    Color? assistantBubbleBorder,
    Color? sectionCardBackground,
    Color? sectionCardBorder,
    Color? chipSelectedBackground,
    Color? chipSelectedText,
    Color? chipSelectedBorder,
    Color? badgeSurface,
    Color? contextLow,
    Color? contextMedium,
    Color? contextHigh,
    Color? gpuTagBackground,
    Color? gpuTagForeground,
    Color? gpuTagBorder,
    Color? cpuTagBackground,
    Color? cpuTagForeground,
    Color? cpuTagBorder,
    Color? hybridTagBackground,
    Color? hybridTagForeground,
    Color? hybridTagBorder,
    Color? neutralTagBackground,
    Color? neutralTagForeground,
    Color? neutralTagBorder,
  }) {
    return AppThemeColors(
      accent: accent ?? this.accent,
      linkColor: linkColor ?? this.linkColor,
      success: success ?? this.success,
      warning: warning ?? this.warning,
      danger: danger ?? this.danger,
      infoSurface: infoSurface ?? this.infoSurface,
      infoBorder: infoBorder ?? this.infoBorder,
      infoIcon: infoIcon ?? this.infoIcon,
      infoText: infoText ?? this.infoText,
      infoSubtext: infoSubtext ?? this.infoSubtext,
      audioAccent: audioAccent ?? this.audioAccent,
      audioSurface: audioSurface ?? this.audioSurface,
      documentAccent: documentAccent ?? this.documentAccent,
      documentSurface: documentSurface ?? this.documentSurface,
      userBubbleBackground: userBubbleBackground ?? this.userBubbleBackground,
      userBubbleBorder: userBubbleBorder ?? this.userBubbleBorder,
      assistantBubbleBackground:
          assistantBubbleBackground ?? this.assistantBubbleBackground,
      assistantBubbleBorder:
          assistantBubbleBorder ?? this.assistantBubbleBorder,
      sectionCardBackground:
          sectionCardBackground ?? this.sectionCardBackground,
      sectionCardBorder: sectionCardBorder ?? this.sectionCardBorder,
      chipSelectedBackground:
          chipSelectedBackground ?? this.chipSelectedBackground,
      chipSelectedText: chipSelectedText ?? this.chipSelectedText,
      chipSelectedBorder: chipSelectedBorder ?? this.chipSelectedBorder,
      badgeSurface: badgeSurface ?? this.badgeSurface,
      contextLow: contextLow ?? this.contextLow,
      contextMedium: contextMedium ?? this.contextMedium,
      contextHigh: contextHigh ?? this.contextHigh,
      gpuTagBackground: gpuTagBackground ?? this.gpuTagBackground,
      gpuTagForeground: gpuTagForeground ?? this.gpuTagForeground,
      gpuTagBorder: gpuTagBorder ?? this.gpuTagBorder,
      cpuTagBackground: cpuTagBackground ?? this.cpuTagBackground,
      cpuTagForeground: cpuTagForeground ?? this.cpuTagForeground,
      cpuTagBorder: cpuTagBorder ?? this.cpuTagBorder,
      hybridTagBackground: hybridTagBackground ?? this.hybridTagBackground,
      hybridTagForeground: hybridTagForeground ?? this.hybridTagForeground,
      hybridTagBorder: hybridTagBorder ?? this.hybridTagBorder,
      neutralTagBackground: neutralTagBackground ?? this.neutralTagBackground,
      neutralTagForeground: neutralTagForeground ?? this.neutralTagForeground,
      neutralTagBorder: neutralTagBorder ?? this.neutralTagBorder,
    );
  }

  @override
  AppThemeColors lerp(covariant ThemeExtension<AppThemeColors>? other, double t) {
    if (other is! AppThemeColors) return this;
    return AppThemeColors(
      accent: Color.lerp(accent, other.accent, t)!,
      linkColor: Color.lerp(linkColor, other.linkColor, t)!,
      success: Color.lerp(success, other.success, t)!,
      warning: Color.lerp(warning, other.warning, t)!,
      danger: Color.lerp(danger, other.danger, t)!,
      infoSurface: Color.lerp(infoSurface, other.infoSurface, t)!,
      infoBorder: Color.lerp(infoBorder, other.infoBorder, t)!,
      infoIcon: Color.lerp(infoIcon, other.infoIcon, t)!,
      infoText: Color.lerp(infoText, other.infoText, t)!,
      infoSubtext: Color.lerp(infoSubtext, other.infoSubtext, t)!,
      audioAccent: Color.lerp(audioAccent, other.audioAccent, t)!,
      audioSurface: Color.lerp(audioSurface, other.audioSurface, t)!,
      documentAccent: Color.lerp(documentAccent, other.documentAccent, t)!,
      documentSurface: Color.lerp(documentSurface, other.documentSurface, t)!,
      userBubbleBackground:
          Color.lerp(userBubbleBackground, other.userBubbleBackground, t)!,
      userBubbleBorder:
          Color.lerp(userBubbleBorder, other.userBubbleBorder, t)!,
      assistantBubbleBackground: Color.lerp(
          assistantBubbleBackground, other.assistantBubbleBackground, t)!,
      assistantBubbleBorder:
          Color.lerp(assistantBubbleBorder, other.assistantBubbleBorder, t)!,
      sectionCardBackground:
          Color.lerp(sectionCardBackground, other.sectionCardBackground, t)!,
      sectionCardBorder:
          Color.lerp(sectionCardBorder, other.sectionCardBorder, t)!,
      chipSelectedBackground:
          Color.lerp(chipSelectedBackground, other.chipSelectedBackground, t)!,
      chipSelectedText:
          Color.lerp(chipSelectedText, other.chipSelectedText, t)!,
      chipSelectedBorder:
          Color.lerp(chipSelectedBorder, other.chipSelectedBorder, t)!,
      badgeSurface: Color.lerp(badgeSurface, other.badgeSurface, t)!,
      contextLow: Color.lerp(contextLow, other.contextLow, t)!,
      contextMedium: Color.lerp(contextMedium, other.contextMedium, t)!,
      contextHigh: Color.lerp(contextHigh, other.contextHigh, t)!,
      gpuTagBackground:
          Color.lerp(gpuTagBackground, other.gpuTagBackground, t)!,
      gpuTagForeground:
          Color.lerp(gpuTagForeground, other.gpuTagForeground, t)!,
      gpuTagBorder: Color.lerp(gpuTagBorder, other.gpuTagBorder, t)!,
      cpuTagBackground:
          Color.lerp(cpuTagBackground, other.cpuTagBackground, t)!,
      cpuTagForeground:
          Color.lerp(cpuTagForeground, other.cpuTagForeground, t)!,
      cpuTagBorder: Color.lerp(cpuTagBorder, other.cpuTagBorder, t)!,
      hybridTagBackground:
          Color.lerp(hybridTagBackground, other.hybridTagBackground, t)!,
      hybridTagForeground:
          Color.lerp(hybridTagForeground, other.hybridTagForeground, t)!,
      hybridTagBorder: Color.lerp(hybridTagBorder, other.hybridTagBorder, t)!,
      neutralTagBackground:
          Color.lerp(neutralTagBackground, other.neutralTagBackground, t)!,
      neutralTagForeground:
          Color.lerp(neutralTagForeground, other.neutralTagForeground, t)!,
      neutralTagBorder:
          Color.lerp(neutralTagBorder, other.neutralTagBorder, t)!,
    );
  }
}

/// Application theme definitions providing the default light theme and
/// the Dracula-inspired dark theme.
class AppTheme {
  static const Color _draculaBackground = Color(0xFF282A36);
  static const Color _draculaSurface = Color(0xFF343746);
  static const Color _draculaCurrentLine = Color(0xFF44475A);
  static const Color _draculaAppBar = Color(0xFF21222C);
  static const Color _draculaForeground = Color(0xFFF8F8F2);
  static const Color _draculaSecondaryText = Color(0xFFB6C2E2);
  static const Color _draculaComment = Color(0xFF6272A4);
  static const Color _draculaPurple = Color(0xFFBD93F9);
  static const Color _draculaCyan = Color(0xFF8BE9FD);
  static const Color _draculaGreen = Color(0xFF50FA7B);
  static const Color _draculaRed = Color(0xFFFF5555);

  static ThemeData get lightTheme {
    final scheme = ColorScheme.fromSeed(seedColor: Colors.teal);
    return ThemeData(
      colorScheme: scheme,
      useMaterial3: true,
      appBarTheme: AppBarTheme(
        backgroundColor: scheme.inversePrimary,
        foregroundColor: scheme.onSurface,
      ),
      extensions: [AppThemeColors.light],
    );
  }

  static ThemeData get darkTheme {
    const scheme = ColorScheme(
      brightness: Brightness.dark,
      primary: _draculaPurple,
      onPrimary: _draculaBackground,
      primaryContainer: _draculaCurrentLine,
      onPrimaryContainer: _draculaForeground,
      secondary: _draculaCyan,
      onSecondary: _draculaBackground,
      secondaryContainer: _draculaCurrentLine,
      onSecondaryContainer: _draculaForeground,
      tertiary: _draculaGreen,
      onTertiary: _draculaBackground,
      tertiaryContainer: Color(0xFF274036),
      onTertiaryContainer: _draculaGreen,
      error: _draculaRed,
      onError: _draculaBackground,
      errorContainer: Color(0xFF4A252B),
      onErrorContainer: Color(0xFFFFC2C2),
      surface: _draculaSurface,
      onSurface: _draculaForeground,
      surfaceContainerLowest: _draculaAppBar,
      surfaceContainerLow: _draculaBackground,
      surfaceContainer: _draculaSurface,
      surfaceContainerHigh: Color(0xFF3D4052),
      surfaceContainerHighest: _draculaCurrentLine,
      onSurfaceVariant: _draculaSecondaryText,
      outline: _draculaComment,
      outlineVariant: _draculaComment,
      shadow: Colors.black,
      scrim: Colors.black,
      inverseSurface: _draculaForeground,
      onInverseSurface: _draculaBackground,
      inversePrimary: _draculaComment,
    );

    final baseDark = ThemeData.dark(useMaterial3: true);
    final subtleBorder = _draculaComment.withValues(alpha: 0.45);

    return baseDark.copyWith(
      colorScheme: scheme,
      scaffoldBackgroundColor: _draculaBackground,
      cardColor: _draculaSurface,
      appBarTheme: const AppBarTheme(
        backgroundColor: _draculaAppBar,
        foregroundColor: _draculaForeground,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        shape: Border(
          bottom: BorderSide(color: _draculaCurrentLine, width: 1),
        ),
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        color: _draculaSurface,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: subtleBorder),
        ),
      ),
      drawerTheme: const DrawerThemeData(
        backgroundColor: _draculaBackground,
        surfaceTintColor: Colors.transparent,
      ),
      dialogTheme: DialogThemeData(
        backgroundColor: _draculaSurface,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
          side: BorderSide(color: subtleBorder),
        ),
      ),
      bottomSheetTheme: const BottomSheetThemeData(
        backgroundColor: _draculaSurface,
        surfaceTintColor: Colors.transparent,
        dragHandleColor: _draculaComment,
      ),
      popupMenuTheme: PopupMenuThemeData(
        color: _draculaSurface,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
          side: BorderSide(color: subtleBorder),
        ),
      ),
      dividerTheme: DividerThemeData(
        color: _draculaComment.withValues(alpha: 0.35),
        thickness: 1,
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: _draculaSurface,
        hintStyle: const TextStyle(color: _draculaSecondaryText),
        labelStyle: const TextStyle(color: _draculaSecondaryText),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: subtleBorder),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: subtleBorder),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: _draculaPurple, width: 1.5),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: _draculaPurple,
          foregroundColor: _draculaBackground,
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: _draculaForeground,
          side: BorderSide(color: _draculaComment.withValues(alpha: 0.65)),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: _draculaCyan,
        ),
      ),
      segmentedButtonTheme: SegmentedButtonThemeData(
        style: ButtonStyle(
          backgroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.selected)) {
              return _draculaPurple.withValues(alpha: 0.24);
            }
            return _draculaBackground;
          }),
          foregroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.selected)) {
              return _draculaForeground;
            }
            return _draculaSecondaryText;
          }),
          side: WidgetStatePropertyAll(
            BorderSide(color: _draculaComment.withValues(alpha: 0.5)),
          ),
        ),
      ),
      snackBarTheme: const SnackBarThemeData(
        backgroundColor: _draculaCurrentLine,
        contentTextStyle: TextStyle(color: _draculaForeground),
        actionTextColor: _draculaCyan,
      ),
      checkboxTheme: CheckboxThemeData(
        fillColor: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return _draculaPurple;
          }
          return Colors.transparent;
        }),
        checkColor: const WidgetStatePropertyAll(_draculaBackground),
        side: const BorderSide(color: _draculaSecondaryText, width: 1.5),
      ),
      extensions: [AppThemeColors.dracula],
    );
  }
}
