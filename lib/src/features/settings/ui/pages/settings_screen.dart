import 'package:flutter/material.dart';
import 'package:get_it/get_it.dart';
import 'package:go_router/go_router.dart';
import 'package:my_hero/src/core/routes/app_routes.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';
import 'package:my_hero/src/core/theme/theme_controller.dart';
import 'package:my_hero/src/features/settings/controller/settings_controller.dart';
import 'package:my_hero/src/features/settings/data/models/settings_args.dart';
import 'package:my_hero/src/features/settings/ui/widgets/appearance_section.dart';
import 'package:my_hero/src/features/settings/ui/widgets/context_window_section.dart';
import 'package:my_hero/src/features/settings/ui/widgets/ollama_model_section.dart';
import 'package:my_hero/src/features/settings/ui/widgets/ollama_server_section.dart';
import 'package:signals_flutter/signals_flutter.dart';

/// Full-page settings screen centralizing model, context window, server, and appearance controls.
class SettingsScreen extends StatefulWidget {
  final SettingsArgs? initialArgs;
  final SettingsController? controller;
  final ThemeController? themeController;

  const SettingsScreen({
    super.key,
    this.initialArgs,
    this.controller,
    this.themeController,
  });

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final SettingsController _controller;
  ThemeController? _themeController;
  final Signal<ThemeMode> _fallbackThemeMode = signal<ThemeMode>(ThemeMode.light);
  late final TextEditingController _hostController;
  late final TextEditingController _pullModelController;
  bool _isSaving = false;

  @override
  void initState() {
    super.initState();
    _controller = widget.controller ?? GetIt.I<SettingsController>();
    _themeController = widget.themeController ??
        (GetIt.I.isRegistered<ThemeController>()
            ? GetIt.I<ThemeController>()
            : null);
    _hostController = TextEditingController();
    _pullModelController = TextEditingController();
    _controller.init(widget.initialArgs).then((_) {
      if (mounted) {
        _hostController.text = _controller.ollamaBaseUrl.value;
      }
    });
    _hostController.text = _controller.ollamaBaseUrl.value;
  }

  @override
  void dispose() {
    _controller.dispose();
    _hostController.dispose();
    _pullModelController.dispose();
    super.dispose();
  }

  void _navigateBack([SettingsResult? result]) {
    if (!mounted) return;
    final goRouter = GoRouter.maybeOf(context);
    if (goRouter != null) {
      if (goRouter.canPop()) {
        goRouter.pop(result);
      } else {
        goRouter.go(AppRoutes.initial);
      }
    } else if (Navigator.of(context).canPop()) {
      Navigator.of(context).pop(result);
    }
  }

  void _handleCancel() {
    _controller.discardChanges();
    _navigateBack();
  }

  Future<void> _handleSave() async {
    if (_isSaving) return;
    setState(() => _isSaving = true);

    try {
      final result = await _controller.saveSettings(
        pendingHostInput: _hostController.text,
      );
      _navigateBack(result);
    } finally {
      if (mounted) {
        setState(() => _isSaving = false);
      }
    }
  }

  void _handleThemeModeChanged(ThemeMode mode) {
    final controller = _themeController;
    if (controller != null) {
      controller.setThemeMode(mode);
    } else {
      _fallbackThemeMode.value = mode;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final appColors = AppThemeColors.of(context);

    return PopScope(
      onPopInvokedWithResult: (didPop, result) {
        if (didPop && result == null) {
          _controller.discardChanges();
        }
      },
      child: Scaffold(
        appBar: AppBar(
          leading: IconButton(
            icon: const Icon(Icons.arrow_back),
            tooltip: 'Voltar',
            onPressed: _isSaving ? null : _handleCancel,
          ),
          title: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.tune, color: appColors.accent),
              const SizedBox(width: 8),
              const Flexible(
                child: Text(
                  'Configurações da Conversa',
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
        ),
        body: Column(
          children: [
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(16.0),
                child: Center(
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 720),
                    child: SignalBuilder(
                      builder: (context) {
                        final isConnected = _controller.isOllamaConnected.value;
                        final serverUrl = _controller.ollamaBaseUrl.value;
                        final models = _controller.availableModels.value;
                        final selectedModel =
                            _controller.selectedModelName.value;
                        final isRefreshing =
                            _controller.isRefreshingModels.value;
                        final isLoadingModel = _controller.isLoadingModel.value;
                        final loadingProgress =
                            _controller.loadingProgress.value;
                        final loadingStatus = _controller.loadingStatus.value;
                        final loadError = _controller.loadErrorMessage.value;
                        final isModelLoaded = _controller.isModelLoaded.value;
                        final downloadState = _controller.downloadState.value;
                        final selectedContextMode =
                            _controller.contextMode.value;
                        final manualContextLength =
                            _controller.manualContextLength.value;
                        final maxModelContext =
                            _controller.activeModelMaxContext;
                        final isEditingServer =
                            _controller.isEditingServer.value;
                        final isTesting =
                            _controller.isTestingConnection.value;
                        final isApplying = _controller.isApplyingServer.value;
                        final testSuccess =
                            _controller.connectionTestSuccess.value;
                        final testMessage =
                            _controller.connectionTestMessage.value;
                        final selectedThemeMode =
                            _themeController?.themeMode.value ??
                                _fallbackThemeMode.value;

                        return Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            OllamaModelSection(
                              isConnected: isConnected,
                              serverUrl: serverUrl,
                              models: models,
                              selectedModelName: selectedModel,
                              isRefreshing: isRefreshing,
                              isLoadingModel: isLoadingModel,
                              loadingProgress: loadingProgress,
                              loadingStatus: loadingStatus,
                              loadErrorMessage: loadError,
                              isModelLoaded: isModelLoaded,
                              downloadState: downloadState,
                              pullModelController: _pullModelController,
                              onRefreshModels: _controller.refreshModels,
                              onModelSelected: _controller.selectModel,
                              onLoadModel: _controller.loadModelIntoMemory,
                              onDownloadModel: _controller.downloadModel,
                              onCancelDownload: _controller.cancelModelDownload,
                            ),
                            const SizedBox(height: 12),
                            AppearanceSection(
                              selectedThemeMode: selectedThemeMode,
                              onThemeModeChanged: _handleThemeModeChanged,
                            ),
                            const SizedBox(height: 12),
                            ContextWindowSection(
                              selectedMode: selectedContextMode,
                              manualContextLength: manualContextLength,
                              maxModelContext: maxModelContext,
                              onModeChanged: _controller.selectContextMode,
                              onManualPresetSelected:
                                  _controller.selectManualContextLength,
                            ),
                            const SizedBox(height: 12),
                            OllamaServerSection(
                              serverUrl: serverUrl,
                              isConnected: isConnected,
                              isEditing: isEditingServer,
                              isTesting: isTesting,
                              isApplying: isApplying,
                              testSuccess: testSuccess,
                              testMessage: testMessage,
                              hostController: _hostController,
                              onToggleEditor: () {
                                if (!isEditingServer) {
                                  _hostController.text = serverUrl;
                                }
                                _controller.toggleServerEditor();
                              },
                              onHostTextChanged: _controller.clearConnectionTest,
                              onSelectLocalhost: _controller.clearConnectionTest,
                              onTestConnection: () => _controller
                                  .testConnection(_hostController.text),
                              onApplyHost: () async {
                                await _controller
                                    .applyServerHost(_hostController.text);
                                if (mounted) {
                                  _hostController.text =
                                      _controller.ollamaBaseUrl.value;
                                }
                              },
                            ),
                          ],
                        );
                      },
                    ),
                  ),
                ),
              ),
            ),
            _buildFooter(theme),
          ],
        ),
      ),
    );
  }

  Widget _buildFooter(ThemeData theme) {
    return SignalBuilder(
      builder: (context) {
        final isBusy = _isSaving ||
            _controller.isLoadingModel.value ||
            _controller.isApplyingServer.value;

        return Container(
          decoration: BoxDecoration(
            color: theme.scaffoldBackgroundColor,
            border: Border(
              top: BorderSide(
                color: theme.colorScheme.outlineVariant.withValues(alpha: 0.4),
              ),
            ),
          ),
          child: SafeArea(
            top: false,
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 720),
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16.0,
                    vertical: 12.0,
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: OutlinedButton(
                          onPressed: isBusy ? null : _handleCancel,
                          style: OutlinedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                          child: const Text('Cancelar'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: FilledButton(
                          onPressed: isBusy ? null : _handleSave,
                          style: FilledButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                          child: _isSaving
                              ? SizedBox(
                                  width: 18,
                                  height: 18,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: theme.colorScheme.onPrimary,
                                  ),
                                )
                              : const Text('Salvar'),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}
