import 'package:flutter/material.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';

/// Card section allowing selection between dynamic automatic `num_ctx` scaling
/// and manual fixed context presets limited by the active model capacity.
class ContextWindowSection extends StatelessWidget {
  final ContextMode selectedMode;
  final int manualContextLength;
  final int maxModelContext;
  final ValueChanged<ContextMode> onModeChanged;
  final ValueChanged<int> onManualPresetSelected;

  const ContextWindowSection({
    super.key,
    required this.selectedMode,
    required this.manualContextLength,
    required this.maxModelContext,
    required this.onModeChanged,
    required this.onManualPresetSelected,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final appColors = AppThemeColors.of(context);
    final allowedPresets = buildContextPresets(maxModelContext);

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
                Icon(Icons.data_usage_rounded, color: appColors.accent),
                const SizedBox(width: 8),
                const Expanded(
                  child: Text(
                    'Janela de Contexto',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            SegmentedButton<ContextMode>(
              segments: [
                ButtonSegment<ContextMode>(
                  value: ContextMode.automatic,
                  label: Text(ContextMode.automatic.label),
                  icon: const Icon(Icons.auto_fix_high_outlined),
                ),
                ButtonSegment<ContextMode>(
                  value: ContextMode.manual,
                  label: Text(ContextMode.manual.label),
                  icon: const Icon(Icons.tune),
                ),
              ],
              selected: {selectedMode},
              onSelectionChanged: (newSelection) {
                onModeChanged(newSelection.first);
              },
            ),
            const SizedBox(height: 10),
            Text(
              selectedMode.description,
              style: TextStyle(
                fontSize: 12,
                color: appColors.infoSubtext,
                fontStyle: FontStyle.italic,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Máximo nativo do modelo ativo: ${ContextSettings.formatTokenCount(maxModelContext)} tokens',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            if (selectedMode == ContextMode.manual) ...[
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: allowedPresets.map((preset) {
                  final isSelected = preset == manualContextLength;
                  final isDefault =
                      preset == ContextSettings.fallbackMaxContext;
                  return Tooltip(
                    message: isDefault
                        ? 'Padrão do Ollama (4.096 tokens)'
                        : '${ContextSettings.formatTokenCount(preset)} tokens',
                    child: ChoiceChip(
                      label: isDefault
                          ? Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Text(preset.toString()),
                                const SizedBox(width: 4),
                                Text(
                                  '(Padrão)',
                                  style: TextStyle(
                                    fontSize: 11,
                                    fontWeight: FontWeight.w500,
                                    color: isSelected
                                        ? appColors.chipSelectedText
                                            .withValues(alpha: 0.85)
                                        : theme.colorScheme.onSurfaceVariant,
                                  ),
                                ),
                              ],
                            )
                          : Text(preset.toString()),
                      selected: isSelected,
                      selectedColor: appColors.chipSelectedBackground,
                      labelStyle: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: isSelected
                            ? appColors.chipSelectedText
                            : theme.colorScheme.onSurface,
                      ),
                      side: BorderSide(
                        color: isSelected
                            ? appColors.chipSelectedBorder
                            : theme.colorScheme.outlineVariant,
                      ),
                      onSelected: (_) => onManualPresetSelected(preset),
                    ),
                  );
                }).toList(),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
