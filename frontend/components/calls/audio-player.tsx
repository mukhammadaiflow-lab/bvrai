"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
  Download,
  Maximize2,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { cn, formatDuration } from "@/lib/utils";

interface AudioPlayerProps {
  src?: string;
  title?: string;
  onTimeUpdate?: (time: number) => void;
  currentTime?: number;
  className?: string;
}

export function AudioPlayer({
  src,
  title,
  onTimeUpdate,
  currentTime: externalTime,
  className,
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);

  // Update time when external time changes (for transcript sync)
  useEffect(() => {
    if (externalTime !== undefined && audioRef.current && Math.abs(audioRef.current.currentTime - externalTime) > 0.5) {
      audioRef.current.currentTime = externalTime;
    }
  }, [externalTime]);

  const togglePlay = useCallback(() => {
    if (!audioRef.current || !src) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying, src]);

  const handleTimeUpdate = useCallback(() => {
    if (!audioRef.current) return;
    const time = audioRef.current.currentTime;
    setCurrentTime(time);
    onTimeUpdate?.(time);
  }, [onTimeUpdate]);

  const handleLoadedMetadata = useCallback(() => {
    if (!audioRef.current) return;
    setDuration(audioRef.current.duration);
    setIsLoading(false);
  }, []);

  const handleSeek = useCallback((value: number[]) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = value[0];
    setCurrentTime(value[0]);
  }, []);

  const handleVolumeChange = useCallback((value: number[]) => {
    if (!audioRef.current) return;
    const newVolume = value[0];
    audioRef.current.volume = newVolume;
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
  }, []);

  const toggleMute = useCallback(() => {
    if (!audioRef.current) return;
    if (isMuted) {
      audioRef.current.volume = volume || 1;
      setIsMuted(false);
    } else {
      audioRef.current.volume = 0;
      setIsMuted(true);
    }
  }, [isMuted, volume]);

  const skip = useCallback((seconds: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = Math.max(0, Math.min(duration, audioRef.current.currentTime + seconds));
  }, [duration]);

  const changePlaybackRate = useCallback(() => {
    if (!audioRef.current) return;
    const rates = [0.5, 0.75, 1, 1.25, 1.5, 2];
    const currentIndex = rates.indexOf(playbackRate);
    const nextIndex = (currentIndex + 1) % rates.length;
    const newRate = rates[nextIndex];
    audioRef.current.playbackRate = newRate;
    setPlaybackRate(newRate);
  }, [playbackRate]);

  const handleDownload = useCallback(() => {
    if (!src) return;
    const a = document.createElement("a");
    a.href = src;
    a.download = title || "recording.mp3";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [src, title]);

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  if (!src) {
    return (
      <div className={cn("rounded-lg border bg-muted/50 p-6 text-center", className)}>
        <VolumeX className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">No recording available</p>
      </div>
    );
  }

  return (
    <div className={cn("rounded-lg border bg-card p-4", className)}>
      <audio
        ref={audioRef}
        src={src}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={() => setIsPlaying(false)}
        onWaiting={() => setIsLoading(true)}
        onCanPlay={() => setIsLoading(false)}
      />

      {/* Title */}
      {title && (
        <div className="mb-3">
          <p className="text-sm font-medium truncate">{title}</p>
        </div>
      )}

      {/* Progress Bar */}
      <div className="mb-3">
        <Slider
          value={[currentTime]}
          max={duration || 100}
          step={0.1}
          onValueChange={handleSeek}
          className="cursor-pointer"
        />
        <div className="flex justify-between mt-1">
          <span className="text-xs text-muted-foreground">{formatDuration(currentTime)}</span>
          <span className="text-xs text-muted-foreground">{formatDuration(duration)}</span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon-sm" onClick={() => skip(-10)} title="Back 10s">
            <SkipBack className="h-4 w-4" />
          </Button>
          <Button
            variant="default"
            size="icon"
            onClick={togglePlay}
            disabled={isLoading}
            className="h-10 w-10"
          >
            {isLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : isPlaying ? (
              <Pause className="h-5 w-5" />
            ) : (
              <Play className="h-5 w-5 ml-0.5" />
            )}
          </Button>
          <Button variant="ghost" size="icon-sm" onClick={() => skip(10)} title="Forward 10s">
            <SkipForward className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center gap-2">
          {/* Playback Rate */}
          <Button
            variant="ghost"
            size="sm"
            onClick={changePlaybackRate}
            className="text-xs font-mono"
          >
            {playbackRate}x
          </Button>

          {/* Volume */}
          <div className="hidden sm:flex items-center gap-1">
            <Button variant="ghost" size="icon-sm" onClick={toggleMute}>
              {isMuted ? (
                <VolumeX className="h-4 w-4" />
              ) : (
                <Volume2 className="h-4 w-4" />
              )}
            </Button>
            <Slider
              value={[isMuted ? 0 : volume]}
              max={1}
              step={0.1}
              onValueChange={handleVolumeChange}
              className="w-20"
            />
          </div>

          {/* Download */}
          <Button variant="ghost" size="icon-sm" onClick={handleDownload} title="Download">
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}

// Compact version for inline use
export function AudioPlayerCompact({
  src,
  onTimeUpdate,
  className,
}: {
  src?: string;
  onTimeUpdate?: (time: number) => void;
  className?: string;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const togglePlay = () => {
    if (!audioRef.current || !src) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  if (!src) {
    return null;
  }

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <audio
        ref={audioRef}
        src={src}
        onTimeUpdate={() => {
          if (audioRef.current) {
            setCurrentTime(audioRef.current.currentTime);
            onTimeUpdate?.(audioRef.current.currentTime);
          }
        }}
        onLoadedMetadata={() => {
          if (audioRef.current) {
            setDuration(audioRef.current.duration);
          }
        }}
        onEnded={() => setIsPlaying(false)}
      />
      <Button variant="ghost" size="icon-sm" onClick={togglePlay}>
        {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
      </Button>
      <span className="text-xs text-muted-foreground">
        {formatDuration(currentTime)} / {formatDuration(duration)}
      </span>
    </div>
  );
}
