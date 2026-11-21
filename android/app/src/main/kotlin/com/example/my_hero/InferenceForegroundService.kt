package com.example.my_hero

import android.Manifest
import android.app.ActivityManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.graphics.drawable.Icon
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log

/**
 * Foreground Service no mesmo processo que mantém a geração de respostas
 * (incluindo Thinking Mode) ativa quando o usuário minimiza o app ou bloqueia
 * a tela no Android.
 *
 * Usa `shortService` no Android 14+ (API 34+), compatível com `targetSdk 36`
 * e sem exigir permissões restritivas adicionais.
 */
class InferenceForegroundService : Service() {

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        ensureChannels(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP_FROM_NOTIFICATION -> {
                isRunning = false
                releaseWakeLock()
                stopForeground(STOP_FOREGROUND_REMOVE)
                onStopRequestedListener?.invoke()
                stopSelf(startId)
                return START_NOT_STICKY
            }

            ACTION_START, null -> {
                val title = intent?.getStringExtra(EXTRA_TITLE) ?: DEFAULT_TITLE
                val body = intent?.getStringExtra(EXTRA_BODY) ?: DEFAULT_BODY

                ensureChannels(this)
                clearCompletedNotification(this)
                acquireWakeLock()

                val notification = buildProgressNotification(this, title, body)
                try {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                        startForeground(
                            PROGRESS_NOTIFICATION_ID,
                            notification,
                            ServiceInfo.FOREGROUND_SERVICE_TYPE_SHORT_SERVICE,
                        )
                    } else {
                        startForeground(PROGRESS_NOTIFICATION_ID, notification)
                    }
                    isRunning = true
                } catch (e: Exception) {
                    Log.w(TAG, "Não foi possível iniciar foreground service: ${e.message}")
                    isRunning = false
                    releaseWakeLock()
                    stopSelf(startId)
                }
                return START_NOT_STICKY
            }

            else -> return START_NOT_STICKY
        }
    }

    override fun onTimeout(startId: Int) {
        Log.w(TAG, "shortService atingiu o limite de tempo do Android; encerrando serviço.")
        isRunning = false
        releaseWakeLock()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf(startId)
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        onTimeout(startId)
    }

    override fun onDestroy() {
        isRunning = false
        releaseWakeLock()
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    private fun acquireWakeLock() {
        try {
            if (wakeLock?.isHeld == true) return
            val powerManager = getSystemService(Context.POWER_SERVICE) as? PowerManager ?: return
            wakeLock = powerManager.newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK,
                WAKELOCK_TAG,
            ).apply {
                setReferenceCounted(false)
                // Timeout de 3 minutos alinhado ao limite do shortService (evita leak de bateria).
                acquire(WAKELOCK_TIMEOUT_MS)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Não foi possível adquirir WakeLock: ${e.message}")
        }
    }

    private fun releaseWakeLock() {
        try {
            if (wakeLock?.isHeld == true) {
                wakeLock?.release()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Erro ao liberar WakeLock: ${e.message}")
        } finally {
            wakeLock = null
        }
    }

    companion object {
        private const val TAG = "InferenceFgService"
        private const val WAKELOCK_TAG = "my_hero:inference_wakelock"
        private const val WAKELOCK_TIMEOUT_MS = 3 * 60 * 1000L

        const val PROGRESS_CHANNEL_ID = "my_hero_inference_progress"
        const val COMPLETE_CHANNEL_ID = "my_hero_inference_complete"

        const val PROGRESS_NOTIFICATION_ID = 4201
        const val COMPLETE_NOTIFICATION_ID = 4202

        const val ACTION_START = "com.example.my_hero.action.START_INFERENCE"
        const val ACTION_STOP_FROM_NOTIFICATION = "com.example.my_hero.action.STOP_INFERENCE"

        const val EXTRA_TITLE = "extra_title"
        const val EXTRA_BODY = "extra_body"

        private const val DEFAULT_TITLE = "My Hero"
        private const val DEFAULT_BODY = "Gerando resposta..."

        @Volatile
        var isRunning: Boolean = false
            private set

        @Volatile
        var onStopRequestedListener: (() -> Unit)? = null

        fun start(context: Context, title: String, body: String) {
            ensureChannels(context)
            clearCompletedNotification(context)

            if (isRunning) {
                updateNotification(context, title, body)
                return
            }

            val intent = Intent(context, InferenceForegroundService::class.java).apply {
                action = ACTION_START
                putExtra(EXTRA_TITLE, title)
                putExtra(EXTRA_BODY, body)
            }
            try {
                context.startForegroundService(intent)
            } catch (e: Exception) {
                Log.w(TAG, "Falha ao disparar InferenceForegroundService: ${e.message}")
            }
        }

        fun updateNotification(context: Context, title: String, body: String) {
            if (!isRunning) return
            if (!canPostNotifications(context)) return

            ensureChannels(context)
            val manager = context.getSystemService(NotificationManager::class.java) ?: return
            manager.notify(
                PROGRESS_NOTIFICATION_ID,
                buildProgressNotification(context, title, body),
            )
        }

        fun stopAndMaybeNotify(
            context: Context,
            notifyIfBackgrounded: Boolean,
            completedTitle: String?,
            completedBody: String?,
        ) {
            isRunning = false
            try {
                context.stopService(Intent(context, InferenceForegroundService::class.java))
            } catch (e: Exception) {
                Log.w(TAG, "Erro ao encerrar InferenceForegroundService: ${e.message}")
            }

            val manager = context.getSystemService(NotificationManager::class.java)
            manager?.cancel(PROGRESS_NOTIFICATION_ID)

            if (notifyIfBackgrounded && !isAppInteractiveAndForeground(context)) {
                showCompletedNotification(
                    context = context,
                    title = completedTitle ?: "Resposta pronta",
                    body = completedBody ?: "Toque para visualizar a resposta no chat.",
                )
            }
        }

        fun clearCompletedNotification(context: Context) {
            val manager = context.getSystemService(NotificationManager::class.java) ?: return
            manager.cancel(COMPLETE_NOTIFICATION_ID)
        }

        private fun showCompletedNotification(
            context: Context,
            title: String,
            body: String,
        ) {
            if (!canPostNotifications(context)) return
            ensureChannels(context)

            val manager = context.getSystemService(NotificationManager::class.java) ?: return
            val openIntent = buildOpenAppPendingIntent(context, requestCode = 2002)

            val notification = Notification.Builder(context, COMPLETE_CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_stat_done)
                .setContentTitle(title)
                .setContentText(body)
                .setStyle(Notification.BigTextStyle().bigText(body))
                .setContentIntent(openIntent)
                .setAutoCancel(true)
                .setCategory(Notification.CATEGORY_MESSAGE)
                .build()

            manager.notify(COMPLETE_NOTIFICATION_ID, notification)
        }

        private fun buildProgressNotification(
            context: Context,
            title: String,
            body: String,
        ): Notification {
            val openIntent = buildOpenAppPendingIntent(context, requestCode = 2001)
            val stopIntent = Intent(context, InferenceForegroundService::class.java).apply {
                action = ACTION_STOP_FROM_NOTIFICATION
            }
            val stopPendingIntent = PendingIntent.getService(
                context,
                2003,
                stopIntent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            )

            val stopAction = Notification.Action.Builder(
                Icon.createWithResource(context, R.drawable.ic_stat_stop),
                "Parar",
                stopPendingIntent,
            ).build()

            return Notification.Builder(context, PROGRESS_CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_stat_inference)
                .setContentTitle(title)
                .setContentText(body)
                .setContentIntent(openIntent)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setProgress(0, 0, true)
                .setCategory(Notification.CATEGORY_PROGRESS)
                .addAction(stopAction)
                .build()
        }

        private fun buildOpenAppPendingIntent(
            context: Context,
            requestCode: Int,
        ): PendingIntent {
            val intent = Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            return PendingIntent.getActivity(
                context,
                requestCode,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            )
        }

        private fun ensureChannels(context: Context) {
            val manager = context.getSystemService(NotificationManager::class.java) ?: return

            if (manager.getNotificationChannel(PROGRESS_CHANNEL_ID) == null) {
                val progressChannel = NotificationChannel(
                    PROGRESS_CHANNEL_ID,
                    "Geração de Respostas",
                    NotificationManager.IMPORTANCE_LOW,
                ).apply {
                    description = "Mantém o modelo respondendo em segundo plano"
                    setShowBadge(false)
                }
                manager.createNotificationChannel(progressChannel)
            }

            if (manager.getNotificationChannel(COMPLETE_CHANNEL_ID) == null) {
                val completeChannel = NotificationChannel(
                    COMPLETE_CHANNEL_ID,
                    "Respostas Concluídas",
                    NotificationManager.IMPORTANCE_DEFAULT,
                ).apply {
                    description = "Avisa quando o modelo termina uma resposta em segundo plano"
                }
                manager.createNotificationChannel(completeChannel)
            }
        }

        private fun canPostNotifications(context: Context): Boolean {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return true
            return context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) ==
                PackageManager.PERMISSION_GRANTED
        }

        private fun isAppInteractiveAndForeground(context: Context): Boolean {
            val processInfo = ActivityManager.RunningAppProcessInfo()
            ActivityManager.getMyMemoryState(processInfo)
            val isForegroundProcess =
                processInfo.importance == ActivityManager.RunningAppProcessInfo.IMPORTANCE_FOREGROUND

            val powerManager = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
            val isScreenInteractive = powerManager?.isInteractive ?: true

            return isForegroundProcess && isScreenInteractive
        }
    }
}
