"use client";

import React, { useState, useCallback, useMemo } from "react";
import {
  Eye,
  EyeOff,
  AlertCircle,
  CheckCircle,
  Info,
  HelpCircle,
  X,
  Plus,
  Minus,
  ChevronDown,
  ChevronUp,
  Calendar,
  Clock,
  Upload,
  File,
  Trash2,
  GripVertical,
  Search,
  Check,
  Copy,
  RefreshCw,
  Loader2,
} from "lucide-react";

// Types
export interface FormField {
  id: string;
  name: string;
  label: string;
  type: FieldType;
  placeholder?: string;
  description?: string;
  required?: boolean;
  disabled?: boolean;
  hidden?: boolean;
  defaultValue?: any;
  validation?: ValidationRule[];
  options?: SelectOption[];
  min?: number;
  max?: number;
  step?: number;
  rows?: number;
  accept?: string;
  multiple?: boolean;
  maxFiles?: number;
  maxSize?: number;
  showCount?: boolean;
  maxLength?: number;
  prefix?: string;
  suffix?: string;
  conditional?: ConditionalRule;
  className?: string;
}

export type FieldType =
  | "text"
  | "email"
  | "password"
  | "number"
  | "tel"
  | "url"
  | "textarea"
  | "select"
  | "multiselect"
  | "checkbox"
  | "radio"
  | "switch"
  | "date"
  | "time"
  | "datetime"
  | "file"
  | "range"
  | "color"
  | "tags"
  | "json"
  | "code"
  | "rich-text";

export interface SelectOption {
  value: string;
  label: string;
  description?: string;
  icon?: React.ReactNode;
  disabled?: boolean;
}

export interface ValidationRule {
  type: "required" | "email" | "url" | "min" | "max" | "minLength" | "maxLength" | "pattern" | "custom";
  value?: any;
  message: string;
  validator?: (value: any, formData: Record<string, any>) => boolean;
}

export interface ConditionalRule {
  field: string;
  operator: "equals" | "notEquals" | "contains" | "notContains" | "greaterThan" | "lessThan" | "isEmpty" | "isNotEmpty";
  value?: any;
}

export interface FormSection {
  id: string;
  title: string;
  description?: string;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
  fields: FormField[];
}

export interface FormBuilderProps {
  fields?: FormField[];
  sections?: FormSection[];
  values?: Record<string, any>;
  errors?: Record<string, string>;
  onChange?: (values: Record<string, any>) => void;
  onSubmit?: (values: Record<string, any>) => void;
  onValidate?: (field: string, value: any) => string | null;
  submitLabel?: string;
  cancelLabel?: string;
  onCancel?: () => void;
  loading?: boolean;
  disabled?: boolean;
  className?: string;
  layout?: "vertical" | "horizontal";
  columns?: number;
}

// Validation helper
function validateField(
  value: any,
  rules: ValidationRule[],
  formData: Record<string, any>
): string | null {
  for (const rule of rules) {
    switch (rule.type) {
      case "required":
        if (
          value === undefined ||
          value === null ||
          value === "" ||
          (Array.isArray(value) && value.length === 0)
        ) {
          return rule.message;
        }
        break;
      case "email":
        if (value && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
          return rule.message;
        }
        break;
      case "url":
        if (value) {
          try {
            new URL(value);
          } catch {
            return rule.message;
          }
        }
        break;
      case "min":
        if (typeof value === "number" && value < rule.value) {
          return rule.message;
        }
        break;
      case "max":
        if (typeof value === "number" && value > rule.value) {
          return rule.message;
        }
        break;
      case "minLength":
        if (typeof value === "string" && value.length < rule.value) {
          return rule.message;
        }
        break;
      case "maxLength":
        if (typeof value === "string" && value.length > rule.value) {
          return rule.message;
        }
        break;
      case "pattern":
        if (value && !new RegExp(rule.value).test(value)) {
          return rule.message;
        }
        break;
      case "custom":
        if (rule.validator && !rule.validator(value, formData)) {
          return rule.message;
        }
        break;
    }
  }
  return null;
}

// Check conditional visibility
function checkConditional(
  rule: ConditionalRule,
  formData: Record<string, any>
): boolean {
  const fieldValue = formData[rule.field];

  switch (rule.operator) {
    case "equals":
      return fieldValue === rule.value;
    case "notEquals":
      return fieldValue !== rule.value;
    case "contains":
      return String(fieldValue).includes(String(rule.value));
    case "notContains":
      return !String(fieldValue).includes(String(rule.value));
    case "greaterThan":
      return Number(fieldValue) > Number(rule.value);
    case "lessThan":
      return Number(fieldValue) < Number(rule.value);
    case "isEmpty":
      return (
        fieldValue === undefined ||
        fieldValue === null ||
        fieldValue === "" ||
        (Array.isArray(fieldValue) && fieldValue.length === 0)
      );
    case "isNotEmpty":
      return (
        fieldValue !== undefined &&
        fieldValue !== null &&
        fieldValue !== "" &&
        (!Array.isArray(fieldValue) || fieldValue.length > 0)
      );
    default:
      return true;
  }
}

// Text Input Component
function TextInput({
  field,
  value,
  error,
  onChange,
  disabled,
}: {
  field: FormField;
  value: any;
  error?: string;
  onChange: (value: any) => void;
  disabled?: boolean;
}) {
  const [showPassword, setShowPassword] = useState(false);
  const inputType =
    field.type === "password" && showPassword ? "text" : field.type;

  return (
    <div className="relative">
      {field.prefix && (
        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 text-sm">
          {field.prefix}
        </span>
      )}
      <input
        id={field.id}
        name={field.name}
        type={inputType}
        value={value || ""}
        onChange={(e) => onChange(e.target.value)}
        placeholder={field.placeholder}
        disabled={disabled || field.disabled}
        min={field.min}
        max={field.max}
        step={field.step}
        maxLength={field.maxLength}
        className={`w-full px-4 py-2.5 bg-white/5 border rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 transition-all ${
          error
            ? "border-red-500/50 focus:border-red-500"
            : "border-white/10 focus:border-purple-500"
        } ${field.prefix ? "pl-10" : ""} ${
          field.suffix || field.type === "password" ? "pr-10" : ""
        } ${disabled || field.disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      />
      {field.type === "password" && (
        <button
          type="button"
          onClick={() => setShowPassword(!showPassword)}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-300"
        >
          {showPassword ? (
            <EyeOff className="w-4 h-4" />
          ) : (
            <Eye className="w-4 h-4" />
          )}
        </button>
      )}
      {field.suffix && field.type !== "password" && (
        <span className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 text-sm">
          {field.suffix}
        </span>
      )}
      {field.showCount && field.maxLength && (
        <span className="absolute right-3 bottom-2 text-xs text-gray-500">
          {(value || "").length}/{field.maxLength}
        </span>
      )}
    </div>
  );
}

// Textarea Component
function TextareaInput({
  field,
  value,
  error,
  onChange,
  disabled,
}: {
  field: FormField;
  value: any;
  error?: string;
  onChange: (value: any) => void;
  disabled?: boolean;
}) {
  return (
    <div className="relative">
      <textarea
        id={field.id}
        name={field.name}
        value={value || ""}
        onChange={(e) => onChange(e.target.value)}
        placeholder={field.placeholder}
        disabled={disabled || field.disabled}
        rows={field.rows || 4}
        maxLength={field.maxLength}
        className={`w-full px-4 py-2.5 bg-white/5 border rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 resize-none transition-all ${
          error
            ? "border-red-500/50 focus:border-red-500"
            : "border-white/10 focus:border-purple-500"
        } ${disabled || field.disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      />
      {field.showCount && field.maxLength && (
        <span className="absolute right-3 bottom-2 text-xs text-gray-500">
          {(value || "").length}/{field.maxLength}
        </span>
      )}
    </div>
  );
}

// Select Component
function SelectInput({
  field,
  value,
  error,
  onChange,
  disabled,
}: {
  field: FormField;
  value: any;
  error?: string;
  onChange: (value: any) => void;
  disabled?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState("");

  const filteredOptions = useMemo(() => {
    if (!search.trim()) return field.options || [];
    return (field.options || []).filter(
      (opt) =>
        opt.label.toLowerCase().includes(search.toLowerCase()) ||
        opt.value.toLowerCase().includes(search.toLowerCase())
    );
  }, [field.options, search]);

  const selectedOption = (field.options || []).find(
    (opt) => opt.value === value
  );

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => !disabled && !field.disabled && setIsOpen(!isOpen)}
        className={`w-full flex items-center justify-between px-4 py-2.5 bg-white/5 border rounded-lg text-left transition-all ${
          error
            ? "border-red-500/50"
            : isOpen
            ? "border-purple-500 ring-2 ring-purple-500/50"
            : "border-white/10"
        } ${disabled || field.disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        <span
          className={selectedOption ? "text-white" : "text-gray-500"}
        >
          {selectedOption?.label || field.placeholder || "Select..."}
        </span>
        <ChevronDown
          className={`w-4 h-4 text-gray-400 transition-transform ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute z-20 w-full mt-1 bg-[#1a1a2e] border border-white/10 rounded-lg shadow-xl overflow-hidden">
            {(field.options?.length || 0) > 5 && (
              <div className="p-2 border-b border-white/10">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    placeholder="Search..."
                    className="w-full pl-9 pr-3 py-2 bg-white/5 border border-white/10 rounded text-white text-sm placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
                  />
                </div>
              </div>
            )}
            <div className="max-h-60 overflow-y-auto">
              {filteredOptions.length > 0 ? (
                filteredOptions.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => {
                      onChange(option.value);
                      setIsOpen(false);
                      setSearch("");
                    }}
                    disabled={option.disabled}
                    className={`w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-white/5 transition-colors ${
                      value === option.value
                        ? "bg-purple-500/10 text-purple-400"
                        : "text-gray-300"
                    } ${option.disabled ? "opacity-50 cursor-not-allowed" : ""}`}
                  >
                    {option.icon && (
                      <span className="flex-shrink-0">{option.icon}</span>
                    )}
                    <div className="flex-1 min-w-0">
                      <span className="block">{option.label}</span>
                      {option.description && (
                        <span className="block text-xs text-gray-500">
                          {option.description}
                        </span>
                      )}
                    </div>
                    {value === option.value && (
                      <Check className="w-4 h-4 text-purple-400" />
                    )}
                  </button>
                ))
              ) : (
                <div className="px-4 py-8 text-center text-gray-500 text-sm">
                  No options found
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// Multi-select Component
function MultiSelectInput({
  field,
  value,
  error,
  onChange,
  disabled,
}: {
  field: FormField;
  value: any;
  error?: string;
  onChange: (value: any) => void;
  disabled?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const selectedValues: string[] = Array.isArray(value) ? value : [];

  const toggleOption = (optionValue: string) => {
    if (selectedValues.includes(optionValue)) {
      onChange(selectedValues.filter((v) => v !== optionValue));
    } else {
      onChange([...selectedValues, optionValue]);
    }
  };

  const selectedOptions = (field.options || []).filter((opt) =>
    selectedValues.includes(opt.value)
  );

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => !disabled && !field.disabled && setIsOpen(!isOpen)}
        className={`w-full flex items-center justify-between px-4 py-2.5 bg-white/5 border rounded-lg text-left transition-all min-h-[42px] ${
          error
            ? "border-red-500/50"
            : isOpen
            ? "border-purple-500 ring-2 ring-purple-500/50"
            : "border-white/10"
        } ${disabled || field.disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        <div className="flex-1 flex flex-wrap gap-1">
          {selectedOptions.length > 0 ? (
            selectedOptions.map((opt) => (
              <span
                key={opt.value}
                className="inline-flex items-center gap-1 px-2 py-0.5 bg-purple-500/20 text-purple-400 text-xs rounded"
              >
                {opt.label}
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleOption(opt.value);
                  }}
                  className="hover:text-purple-300"
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))
          ) : (
            <span className="text-gray-500">
              {field.placeholder || "Select..."}
            </span>
          )}
        </div>
        <ChevronDown
          className={`w-4 h-4 text-gray-400 transition-transform flex-shrink-0 ml-2 ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute z-20 w-full mt-1 bg-[#1a1a2e] border border-white/10 rounded-lg shadow-xl overflow-hidden max-h-60 overflow-y-auto">
            {(field.options || []).map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => toggleOption(option.value)}
                disabled={option.disabled}
                className={`w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-white/5 transition-colors ${
                  selectedValues.includes(option.value)
                    ? "bg-purple-500/10 text-purple-400"
                    : "text-gray-300"
                } ${option.disabled ? "opacity-50 cursor-not-allowed" : ""}`}
              >
                <div
                  className={`w-4 h-4 rounded border flex items-center justify-center ${
                    selectedValues.includes(option.value)
                      ? "bg-purple-500 border-purple-500"
                      : "border-gray-600"
                  }`}
                >
                  {selectedValues.includes(option.value) && (
                    <Check className="w-3 h-3 text-white" />
                  )}
                </div>
                <span>{option.label}</span>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// Checkbox Component
function CheckboxInput({
  field,
  value,
  error,
  onChange,
  disabled,
}: {
  field: FormField;
  value: any;
  error?: string;
  onChange: (value: any) => void;
  disabled?: boolean;
}) {
  return (
    <label
      className={`flex items-start gap-3 cursor-pointer ${
        disabled || field.disabled ? "opacity-50 cursor-not-allowed" : ""
      }`}
    >
      <div
        onClick={() => !disabled && !field.disabled && onChange(!value)}
        className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-all flex-shrink-0 mt-0.5 ${
          value
            ? "bg-purple-500 border-purple-500"
            : error
            ? "border-red-500/50"
            : "border-gray-600 hover:border-gray-500"
        }`}
      >
        {value && <Check className="w-3 h-3 text-white" />}
      </div>
      <div>
        <span className="text-white">{field.label}</span>
        {field.description && (
          <p className="text-sm text-gray-500 mt-0.5">{field.description}</p>
        )}
      </div>
    </label>
  );
}

// Radio Group Component
function RadioInput({
  field,
  value,
  error,
  onChange,
  disabled,
}: {
  field: FormField;
  value: any;
  error?: string;
  onChange: (value: any) => void;
  disabled?: boolean;
}) {
  return (
    <div className="space-y-2">
      {(field.options || []).map((option) => (
        <label
          key={option.value}
          className={`flex items-start gap-3 cursor-pointer ${
            disabled || field.disabled || option.disabled
              ? "opacity-50 cursor-not-allowed"
              : ""
          }`}
        >
          <div
            onClick={() =>
              !disabled &&
              !field.disabled &&
              !option.disabled &&
              onChange(option.value)
            }
            className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-all flex-shrink-0 mt-0.5 ${
              value === option.value
                ? "border-purple-500"
                : error
                ? "border-red-500/50"
                : "border-gray-600 hover:border-gray-500"
            }`}
          >
            {value === option.value && (
              <div className="w-2.5 h-2.5 rounded-full bg-purple-500" />
            )}
          </div>
          <div>
            <span className="text-white">{option.label}</span>
            {option.description && (
              <p className="text-sm text-gray-500 mt-0.5">
                {option.description}
              </p>
            )}
          </div>
        </label>
      ))}
    </div>
  );
}

// Switch Component
function SwitchInput({
  field,
  value,
  error,
  onChange,
  disabled,
}: {
  field: FormField;
  value: any;
  error?: string;
  onChange: (value: any) => void;
  disabled?: boolean;
}) {
  return (
    <label
      className={`flex items-center justify-between cursor-pointer ${
        disabled || field.disabled ? "opacity-50 cursor-not-allowed" : ""
      }`}
    >
      <div>
        <span className="text-white">{field.label}</span>
        {field.description && (
          <p className="text-sm text-gray-500 mt-0.5">{field.description}</p>
        )}
      </div>
      <button
        type="button"
        onClick={() => !disabled && !field.disabled && onChange(!value)}
        className={`relative w-11 h-6 rounded-full transition-colors ${
          value ? "bg-purple-500" : "bg-gray-600"
        }`}
      >
        <span
          className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
            value ? "translate-x-6" : "translate-x-1"
          }`}
        />
      </button>
    </label>
  );
}

// Range Slider Component
function RangeInput({
  field,
  value,
  error,
  onChange,
  disabled,
}: {
  field: FormField;
  value: any;
  error?: string;
  onChange: (value: any) => void;
  disabled?: boolean;
}) {
  const min = field.min ?? 0;
  const max = field.max ?? 100;
  const currentValue = value ?? min;
  const percentage = ((currentValue - min) / (max - min)) * 100;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-400">{min}</span>
        <span className="text-sm font-medium text-purple-400">
          {currentValue}
        </span>
        <span className="text-sm text-gray-400">{max}</span>
      </div>
      <div className="relative h-2">
        <div className="absolute inset-0 bg-white/10 rounded-full" />
        <div
          className="absolute inset-y-0 left-0 bg-purple-500 rounded-full"
          style={{ width: `${percentage}%` }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={field.step ?? 1}
          value={currentValue}
          onChange={(e) => onChange(Number(e.target.value))}
          disabled={disabled || field.disabled}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-white rounded-full shadow-lg pointer-events-none"
          style={{ left: `calc(${percentage}% - 8px)` }}
        />
      </div>
    </div>
  );
}

// Tags Input Component
function TagsInput({
  field,
  value,
  error,
  onChange,
  disabled,
}: {
  field: FormField;
  value: any;
  error?: string;
  onChange: (value: any) => void;
  disabled?: boolean;
}) {
  const [inputValue, setInputValue] = useState("");
  const tags: string[] = Array.isArray(value) ? value : [];

  const addTag = () => {
    const tag = inputValue.trim();
    if (tag && !tags.includes(tag)) {
      onChange([...tags, tag]);
      setInputValue("");
    }
  };

  const removeTag = (tagToRemove: string) => {
    onChange(tags.filter((t) => t !== tagToRemove));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addTag();
    } else if (e.key === "Backspace" && !inputValue && tags.length > 0) {
      removeTag(tags[tags.length - 1]);
    }
  };

  return (
    <div
      className={`flex flex-wrap gap-2 p-2 bg-white/5 border rounded-lg transition-all min-h-[42px] ${
        error ? "border-red-500/50" : "border-white/10 focus-within:border-purple-500"
      } ${disabled || field.disabled ? "opacity-50" : ""}`}
    >
      {tags.map((tag) => (
        <span
          key={tag}
          className="inline-flex items-center gap-1 px-2 py-1 bg-purple-500/20 text-purple-400 text-sm rounded"
        >
          {tag}
          {!disabled && !field.disabled && (
            <button
              type="button"
              onClick={() => removeTag(tag)}
              className="hover:text-purple-300"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </span>
      ))}
      <input
        type="text"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyDown={handleKeyDown}
        onBlur={addTag}
        placeholder={tags.length === 0 ? field.placeholder : ""}
        disabled={disabled || field.disabled}
        className="flex-1 min-w-[100px] bg-transparent text-white placeholder:text-gray-500 focus:outline-none"
      />
    </div>
  );
}

// File Upload Component
function FileInput({
  field,
  value,
  error,
  onChange,
  disabled,
}: {
  field: FormField;
  value: any;
  error?: string;
  onChange: (value: any) => void;
  disabled?: boolean;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const files: File[] = Array.isArray(value) ? value : value ? [value] : [];

  const handleFiles = (newFiles: FileList | null) => {
    if (!newFiles) return;
    const fileArray = Array.from(newFiles);
    if (field.multiple) {
      onChange([...files, ...fileArray]);
    } else {
      onChange(fileArray[0]);
    }
  };

  const removeFile = (index: number) => {
    if (field.multiple) {
      onChange(files.filter((_, i) => i !== index));
    } else {
      onChange(null);
    }
  };

  return (
    <div className="space-y-2">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={`relative border-2 border-dashed rounded-lg p-6 text-center transition-all ${
          isDragging
            ? "border-purple-500 bg-purple-500/10"
            : error
            ? "border-red-500/50"
            : "border-white/10 hover:border-white/20"
        } ${disabled || field.disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        <input
          type="file"
          accept={field.accept}
          multiple={field.multiple}
          onChange={(e) => handleFiles(e.target.files)}
          disabled={disabled || field.disabled}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
        <p className="text-white">
          Drag & drop files or <span className="text-purple-400">browse</span>
        </p>
        <p className="text-sm text-gray-500 mt-1">
          {field.accept || "Any file type"}
          {field.maxSize && ` • Max ${field.maxSize}MB`}
        </p>
      </div>

      {files.length > 0 && (
        <div className="space-y-2">
          {files.map((file, index) => (
            <div
              key={index}
              className="flex items-center gap-3 p-3 bg-white/5 rounded-lg"
            >
              <File className="w-5 h-5 text-gray-400" />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-white truncate">{file.name}</p>
                <p className="text-xs text-gray-500">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <button
                type="button"
                onClick={() => removeFile(index)}
                disabled={disabled || field.disabled}
                className="p-1 hover:bg-white/10 rounded text-gray-400 hover:text-red-400"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Form Field Wrapper
function FormFieldWrapper({
  field,
  value,
  error,
  onChange,
  disabled,
  layout,
}: {
  field: FormField;
  value: any;
  error?: string;
  onChange: (value: any) => void;
  disabled?: boolean;
  layout: "vertical" | "horizontal";
}) {
  // Skip rendering for checkbox/switch/radio as they have built-in labels
  const hasBuiltInLabel = ["checkbox", "switch", "radio"].includes(field.type);

  const renderInput = () => {
    switch (field.type) {
      case "text":
      case "email":
      case "password":
      case "number":
      case "tel":
      case "url":
        return (
          <TextInput
            field={field}
            value={value}
            error={error}
            onChange={onChange}
            disabled={disabled}
          />
        );
      case "textarea":
        return (
          <TextareaInput
            field={field}
            value={value}
            error={error}
            onChange={onChange}
            disabled={disabled}
          />
        );
      case "select":
        return (
          <SelectInput
            field={field}
            value={value}
            error={error}
            onChange={onChange}
            disabled={disabled}
          />
        );
      case "multiselect":
        return (
          <MultiSelectInput
            field={field}
            value={value}
            error={error}
            onChange={onChange}
            disabled={disabled}
          />
        );
      case "checkbox":
        return (
          <CheckboxInput
            field={field}
            value={value}
            error={error}
            onChange={onChange}
            disabled={disabled}
          />
        );
      case "radio":
        return (
          <RadioInput
            field={field}
            value={value}
            error={error}
            onChange={onChange}
            disabled={disabled}
          />
        );
      case "switch":
        return (
          <SwitchInput
            field={field}
            value={value}
            error={error}
            onChange={onChange}
            disabled={disabled}
          />
        );
      case "range":
        return (
          <RangeInput
            field={field}
            value={value}
            error={error}
            onChange={onChange}
            disabled={disabled}
          />
        );
      case "tags":
        return (
          <TagsInput
            field={field}
            value={value}
            error={error}
            onChange={onChange}
            disabled={disabled}
          />
        );
      case "file":
        return (
          <FileInput
            field={field}
            value={value}
            error={error}
            onChange={onChange}
            disabled={disabled}
          />
        );
      default:
        return (
          <TextInput
            field={field}
            value={value}
            error={error}
            onChange={onChange}
            disabled={disabled}
          />
        );
    }
  };

  if (hasBuiltInLabel) {
    return (
      <div className={field.className}>
        {renderInput()}
        {error && (
          <p className="flex items-center gap-1 text-sm text-red-400 mt-1">
            <AlertCircle className="w-3 h-3" />
            {error}
          </p>
        )}
      </div>
    );
  }

  return (
    <div
      className={`${
        layout === "horizontal"
          ? "grid grid-cols-3 gap-4 items-start"
          : "space-y-2"
      } ${field.className || ""}`}
    >
      <label htmlFor={field.id} className="block">
        <span className="text-sm font-medium text-gray-300">
          {field.label}
          {field.required && <span className="text-red-400 ml-1">*</span>}
        </span>
        {field.description && layout === "horizontal" && (
          <p className="text-xs text-gray-500 mt-0.5">{field.description}</p>
        )}
      </label>

      <div className={layout === "horizontal" ? "col-span-2" : ""}>
        {renderInput()}
        {field.description && layout === "vertical" && (
          <p className="text-xs text-gray-500 mt-1">{field.description}</p>
        )}
        {error && (
          <p className="flex items-center gap-1 text-sm text-red-400 mt-1">
            <AlertCircle className="w-3 h-3" />
            {error}
          </p>
        )}
      </div>
    </div>
  );
}

// Main Form Builder Component
export function FormBuilder({
  fields = [],
  sections = [],
  values: controlledValues,
  errors: controlledErrors,
  onChange,
  onSubmit,
  onValidate,
  submitLabel = "Submit",
  cancelLabel = "Cancel",
  onCancel,
  loading = false,
  disabled = false,
  className = "",
  layout = "vertical",
  columns = 1,
}: FormBuilderProps) {
  const [internalValues, setInternalValues] = useState<Record<string, any>>(
    {}
  );
  const [internalErrors, setInternalErrors] = useState<Record<string, string>>(
    {}
  );
  const [collapsedSections, setCollapsedSections] = useState<Set<string>>(
    () => {
      const initial = new Set<string>();
      sections.forEach((section) => {
        if (section.defaultCollapsed) {
          initial.add(section.id);
        }
      });
      return initial;
    }
  );

  const values = controlledValues ?? internalValues;
  const errors = controlledErrors ?? internalErrors;

  // Get all fields from sections or direct fields
  const allFields = useMemo(() => {
    if (sections.length > 0) {
      return sections.flatMap((section) => section.fields);
    }
    return fields;
  }, [sections, fields]);

  // Initialize default values
  React.useEffect(() => {
    const defaults: Record<string, any> = {};
    allFields.forEach((field) => {
      if (field.defaultValue !== undefined && values[field.name] === undefined) {
        defaults[field.name] = field.defaultValue;
      }
    });
    if (Object.keys(defaults).length > 0) {
      handleChange(defaults);
    }
  }, []);

  const handleChange = useCallback(
    (updates: Record<string, any>) => {
      const newValues = { ...values, ...updates };
      if (onChange) {
        onChange(newValues);
      } else {
        setInternalValues(newValues);
      }
    },
    [values, onChange]
  );

  const handleFieldChange = useCallback(
    (fieldName: string, value: any) => {
      handleChange({ [fieldName]: value });

      // Clear error on change
      if (errors[fieldName]) {
        if (controlledErrors === undefined) {
          setInternalErrors((prev) => {
            const newErrors = { ...prev };
            delete newErrors[fieldName];
            return newErrors;
          });
        }
      }
    },
    [handleChange, errors, controlledErrors]
  );

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();

      // Validate all fields
      const newErrors: Record<string, string> = {};
      allFields.forEach((field) => {
        // Check visibility
        if (field.conditional && !checkConditional(field.conditional, values)) {
          return;
        }

        // Validate
        if (field.validation) {
          const error = validateField(values[field.name], field.validation, values);
          if (error) {
            newErrors[field.name] = error;
          }
        }

        // Custom validation
        if (onValidate) {
          const error = onValidate(field.name, values[field.name]);
          if (error) {
            newErrors[field.name] = error;
          }
        }
      });

      if (Object.keys(newErrors).length > 0) {
        if (controlledErrors === undefined) {
          setInternalErrors(newErrors);
        }
        return;
      }

      onSubmit?.(values);
    },
    [allFields, values, onValidate, onSubmit, controlledErrors]
  );

  const toggleSection = (sectionId: string) => {
    setCollapsedSections((prev) => {
      const next = new Set(prev);
      if (next.has(sectionId)) {
        next.delete(sectionId);
      } else {
        next.add(sectionId);
      }
      return next;
    });
  };

  const renderField = (field: FormField) => {
    // Check conditional visibility
    if (field.conditional && !checkConditional(field.conditional, values)) {
      return null;
    }

    if (field.hidden) {
      return null;
    }

    return (
      <FormFieldWrapper
        key={field.id}
        field={field}
        value={values[field.name]}
        error={errors[field.name]}
        onChange={(value) => handleFieldChange(field.name, value)}
        disabled={disabled || loading}
        layout={layout}
      />
    );
  };

  const renderFields = (fieldsToRender: FormField[]) => {
    if (columns > 1) {
      return (
        <div
          className="grid gap-4"
          style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}
        >
          {fieldsToRender.map(renderField)}
        </div>
      );
    }
    return <div className="space-y-4">{fieldsToRender.map(renderField)}</div>;
  };

  return (
    <form onSubmit={handleSubmit} className={`space-y-6 ${className}`}>
      {sections.length > 0
        ? sections.map((section) => (
            <div
              key={section.id}
              className="bg-[#1a1a2e]/30 rounded-xl border border-white/5 overflow-hidden"
            >
              {/* Section header */}
              <div
                onClick={() =>
                  section.collapsible && toggleSection(section.id)
                }
                className={`flex items-center justify-between px-6 py-4 border-b border-white/5 ${
                  section.collapsible ? "cursor-pointer hover:bg-white/5" : ""
                }`}
              >
                <div>
                  <h3 className="font-semibold text-white">{section.title}</h3>
                  {section.description && (
                    <p className="text-sm text-gray-500 mt-0.5">
                      {section.description}
                    </p>
                  )}
                </div>
                {section.collapsible && (
                  <ChevronDown
                    className={`w-5 h-5 text-gray-400 transition-transform ${
                      collapsedSections.has(section.id) ? "-rotate-180" : ""
                    }`}
                  />
                )}
              </div>

              {/* Section content */}
              {!collapsedSections.has(section.id) && (
                <div className="p-6">{renderFields(section.fields)}</div>
              )}
            </div>
          ))
        : renderFields(fields)}

      {/* Submit buttons */}
      {(onSubmit || onCancel) && (
        <div className="flex items-center justify-end gap-3 pt-4 border-t border-white/10">
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              disabled={loading}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
            >
              {cancelLabel}
            </button>
          )}
          {onSubmit && (
            <button
              type="submit"
              disabled={loading || disabled}
              className="flex items-center gap-2 px-6 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading && <Loader2 className="w-4 h-4 animate-spin" />}
              {submitLabel}
            </button>
          )}
        </div>
      )}
    </form>
  );
}

export default FormBuilder;
