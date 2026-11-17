import 'package:flutter/material.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';

/// Card section displaying the segmented appearance selector for light and
/// dark (Dracula) themes.
class AppearanceSection extends StatelessWidget {
  final ThemeMode selectedThemeMode;
  final ValueChanged<ThemeMode> onThemeModeChanged;

  const AppearanceSection({
    super.key,
    required this.selectedThemeMode,
    required this.onThemeModeChanged,
  });

  ThemeMode get _effectiveMode =>
      selectedThemeMode == ThemeMode.dark ? ThemeMode.dark : ThemeMode.light;

  String get _description => _effectiveMode == ThemeMode.dark
      ? 'Tema escuro inspirado na paleta Dracula.'
      : 'Tema claro padrão com tons em verde-água.';

  @override
  Widget build(BuildContext context) {
    final appColors = AppThemeColors.of(context);

    return Card(
      elevation: 0,
      color: appColors.sectionCardBackground,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: appColors.sectionCardBorder),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Icon(Icons.palette_outlined, color: appColors.accent),
                const SizedBox(width: 8),
                const Text(
                  'Aparência',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
                ),
              ],
            ),
            const SizedBox(height: 12),
            SegmentedButton<ThemeMode>(
              showSelectedIcon: false,
              segments: const [
                ButtonSegment<ThemeMode>(
                  value: ThemeMode.light,
                  label: Text('Claro'),
                  icon: Icon(Icons.light_mode_outlined),
                ),
                ButtonSegment<ThemeMode>(
                  value: ThemeMode.dark,
                  label: Text('Escuro'),
                  icon: Icon(Icons.dark_mode_outlined),
                ),
              ],
              selected: {_effectiveMode},
              onSelectionChanged: (newSelection) {
                onThemeModeChanged(newSelection.first);
              },
            ),
            const SizedBox(height: 10),
            Text(
              _description,
              style: TextStyle(
                fontSize: 12,
                color: appColors.infoSubtext,
                fontStyle: FontStyle.italic,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
