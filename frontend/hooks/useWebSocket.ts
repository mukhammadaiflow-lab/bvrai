/**
 * WebSocket Hook
 *
 * Provides real-time connection for live call updates.
 * Handles connection lifecycle and message handling.
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { useAuth } from './useAuth';

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

interface UseWebSocketOptions {
  url?: string;
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  autoReconnect?: boolean;
  reconnectInterval?: number;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    url = `${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'}/ws`,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    autoReconnect = true,
    reconnectInterval = 5000,
  } = options;

  const { token } = useAuth();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const wsUrl = token ? `${url}?token=${token}` : url;
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        setIsConnected(true);
        onConnect?.();
        console.log('WebSocket connected');
      };

      wsRef.current.onclose = () => {
        setIsConnected(false);
        onDisconnect?.();
        console.log('WebSocket disconnected');

        // Auto-reconnect
        if (autoReconnect) {
          reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        onError?.(error);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(message);
          onMessage?.(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
    }
  }, [url, token, onMessage, onConnect, onDisconnect, onError, autoReconnect, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const sendMessage = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected');
    }
  }, []);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    isConnected,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
  };
}

/**
 * Hook for subscribing to specific call updates
 */
export function useCallUpdates(
  callId: string | null,
  onUpdate?: (data: any) => void
) {
  const { sendMessage } = useWebSocket({
    onMessage: (message) => {
      if (message.type === 'call_update' && message.data?.call_id === callId) {
        onUpdate?.(message.data);
      }
    },
  });

  useEffect(() => {
    if (callId) {
      sendMessage({ type: 'subscribe', call_id: callId });
      return () => {
        sendMessage({ type: 'unsubscribe', call_id: callId });
      };
    }
  }, [callId, sendMessage]);
}

/**
 * Hook for monitoring real-time metrics
 */
export function useRealTimeMetrics(onMetricsUpdate?: (metrics: any) => void) {
  const [metrics, setMetrics] = useState<any>(null);

  useWebSocket({
    onMessage: (message) => {
      if (message.type === 'metrics_update') {
        setMetrics(message.data);
        onMetricsUpdate?.(message.data);
      }
    },
  });

  return metrics;
}
