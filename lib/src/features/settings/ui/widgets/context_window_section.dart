import 'package:flutter/material.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';

/// Card section allowing selection between dynamic automatic `num_ctx` scaling
/// and manual fixed context presets limited by the active model capacity.
class ContextWindowSection extends StatelessWidget {
  final ContextMode selectedMode;
  final int manualContextLength;
  final int maxModelContext;
  final double? serverMemoryLimitGb;
  final ValueChanged<ContextMode> onModeChanged;
  final ValueChanged<int> onManualPresetSelected;

  const ContextWindowSection({
    super.key,
    required this.selectedMode,
    required this.manualContextLength,
    required this.maxModelContext,
    required this.onModeChanged,
    required this.onManualPresetSelected,
    this.serverMemoryLimitGb,
  });

  String _formatGb(double gb) {
    return gb == gb.roundToDouble()
        ? gb.toStringAsFixed(0)
        : gb.toStringAsFixed(1);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final allowedPresets =
        ContextSettings.availablePresetsForModel(maxModelContext);

    return Card(
      elevation: 0,
      color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.4),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: theme.colorScheme.outlineVariant.withValues(alpha: 0.5),
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Row(
              children: [
                Icon(Icons.data_usage_rounded, color: Colors.teal),
                SizedBox(width: 8),
                Expanded(
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
                color: Colors.teal.shade800,
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
                  return ChoiceChip(
                    label: Text(preset.toString()),
                    selected: isSelected,
                    selectedColor: Colors.teal.withValues(alpha: 0.18),
                    labelStyle: TextStyle(
                      fontWeight: FontWeight.w600,
                      color: isSelected
                          ? Colors.teal.shade900
                          : theme.colorScheme.onSurface,
                    ),
                    side: BorderSide(
                      color: isSelected
                          ? Colors.teal
                          : theme.colorScheme.outlineVariant,
                    ),
                    onSelected: (_) => onManualPresetSelected(preset),
                  );
                }).toList(),
              ),
            ] else if (serverMemoryLimitGb != null &&
                serverMemoryLimitGb! > 0) ...[
              const SizedBox(height: 10),
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: Colors.teal.shade50,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: Colors.teal.shade200),
                ),
                child: Row(
                  children: [
                    Icon(
                      Icons.shield_outlined,
                      size: 18,
                      color: Colors.teal.shade800,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Escalonamento em degraus limitado pela memória informada (${_formatGb(serverMemoryLimitGb!)} GB).',
                        style: TextStyle(
                          fontSize: 11.5,
                          color: Colors.teal.shade900,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
