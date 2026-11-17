import 'package:flutter/material.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';

/// Card section displaying a discrete power-of-two `num_ctx` slider capped
/// at the active model's native capacity.
class ContextWindowSection extends StatelessWidget {
  final int selectedContextLength;
  final int maxModelContext;
  final ValueChanged<int> onContextLengthChanged;

  const ContextWindowSection({
    super.key,
    required this.selectedContextLength,
    required this.maxModelContext,
    required this.onContextLengthChanged,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final appColors = AppThemeColors.of(context);
    final presets = buildContextPresets(maxModelContext);
    final effectiveLength = resolveClosestContextPreset(
      selectedContextLength,
      presets,
    );
    final selectedIndex = presets.indexOf(effectiveLength);
    final isSliderEnabled = presets.length > 1;

    final compactLabel = formatContextLabel(effectiveLength);
    final exactTokens = ContextSettings.formatTokenCount(effectiveLength);
    final badgeText = effectiveLength == ContextSettings.fallbackMaxContext
        ? '$compactLabel (Padrão) • $exactTokens tokens'
        : '$compactLabel • $exactTokens tokens';

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
            SizedBox(
              width: double.infinity,
              child: Wrap(
                alignment: WrapAlignment.spaceBetween,
                crossAxisAlignment: WrapCrossAlignment.center,
                spacing: 8,
                runSpacing: 8,
                children: [
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.data_usage_rounded, color: appColors.accent),
                      const SizedBox(width: 8),
                      const Flexible(
                        child: Text(
                          'Janela de Contexto',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 15,
                          ),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ),
                  Container(
                    key: const Key('context_window_value_badge'),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 5,
                    ),
                    decoration: BoxDecoration(
                      color: appColors.chipSelectedBackground,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color:
                            appColors.chipSelectedBorder.withValues(alpha: 0.7),
                      ),
                    ),
                    child: Text(
                      badgeText,
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: appColors.chipSelectedText,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            SliderTheme(
              data: SliderTheme.of(context).copyWith(
                overlayShape: const RoundSliderOverlayShape(overlayRadius: 16),
                padding: const EdgeInsets.symmetric(horizontal: 16),
              ),
              child: Slider(
                key: const Key('context_window_slider'),
                min: 0,
                max: isSliderEnabled ? (presets.length - 1).toDouble() : 1,
                divisions: isSliderEnabled ? presets.length - 1 : null,
                value: isSliderEnabled ? selectedIndex.toDouble() : 0,
                label: isSliderEnabled ? compactLabel : null,
                activeColor: theme.colorScheme.primary,
                thumbColor: isSliderEnabled ? theme.colorScheme.primary : null,
                inactiveColor:
                    theme.colorScheme.outlineVariant.withValues(alpha: 0.45),
                onChanged: isSliderEnabled
                    ? (rawValue) {
                        final index =
                            rawValue.round().clamp(0, presets.length - 1);
                        onContextLengthChanged(presets[index]);
                      }
                    : null,
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  _buildEndpointLabel(
                    context: context,
                    tokens: presets.first,
                    isActive: effectiveLength == presets.first,
                    isEnabled: isSliderEnabled,
                  ),
                  if (isSliderEnabled)
                    _buildEndpointLabel(
                      context: context,
                      tokens: presets.last,
                      isActive: effectiveLength == presets.last,
                      isEnabled: true,
                    ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            Text(
              'Máximo nativo do modelo ativo: ${ContextSettings.formatTokenCount(maxModelContext)} tokens',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEndpointLabel({
    required BuildContext context,
    required int tokens,
    required bool isActive,
    required bool isEnabled,
  }) {
    final theme = Theme.of(context);
    final label = formatContextLabel(tokens);

    return Material(
      type: MaterialType.transparency,
      child: InkWell(
        borderRadius: BorderRadius.circular(6),
        onTap: isEnabled ? () => onContextLengthChanged(tokens) : null,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 4),
          child: Text(
            label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: isActive ? FontWeight.bold : FontWeight.w500,
              color: isActive
                  ? theme.colorScheme.primary
                  : theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ),
      ),
    );
  }
}
