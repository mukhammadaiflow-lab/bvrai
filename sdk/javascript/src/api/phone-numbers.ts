/**
 * Phone Numbers API for Builder Engine SDK
 */

import type { HttpClient } from '../utils/http';
import type {
  PhoneNumber,
  AvailableNumber,
  PhoneNumberType,
  PhoneNumberStatus,
  PhoneNumberCapability,
  SearchNumbersRequest,
  RequestOptions,
  PaginatedResponse,
  Metadata,
} from '../types';

export interface PhoneNumberConfig {
  voiceUrl?: string;
  voiceFallbackUrl?: string;
  statusCallbackUrl?: string;
  voiceMethod?: 'POST' | 'GET';
  smsUrl?: string;
  smsFallbackUrl?: string;
  recordingEnabled?: boolean;
  transcriptionEnabled?: boolean;
  callerIdLookup?: boolean;
}

export class PhoneNumbersAPI {
  constructor(private readonly http: HttpClient) {}

  /**
   * List all phone numbers
   */
  async list(
    params?: {
      agentId?: string;
      status?: PhoneNumberStatus;
      limit?: number;
      offset?: number;
    },
    options?: RequestOptions
  ): Promise<PaginatedResponse<PhoneNumber>> {
    const response = await this.http.get<{
      phone_numbers: PhoneNumber[];
      total: number;
      limit: number;
      offset: number;
    }>(
      '/v1/phone-numbers',
      {
        agent_id: params?.agentId,
        status: params?.status,
        limit: params?.limit,
        offset: params?.offset,
      },
      options
    );

    return {
      items: response.phone_numbers.map((n) => this.transformPhoneNumber(n)),
      total: response.total,
      limit: response.limit,
      offset: response.offset,
      hasMore: response.offset + response.phone_numbers.length < response.total,
    };
  }

  /**
   * Get a phone number by ID
   */
  async get(numberId: string, options?: RequestOptions): Promise<PhoneNumber> {
    const response = await this.http.get<PhoneNumber>(
      `/v1/phone-numbers/${numberId}`,
      undefined,
      options
    );
    return this.transformPhoneNumber(response);
  }

  /**
   * Get a phone number by the actual number
   */
  async getByNumber(phoneNumber: string, options?: RequestOptions): Promise<PhoneNumber> {
    const response = await this.http.get<PhoneNumber>(
      `/v1/phone-numbers/lookup/${encodeURIComponent(phoneNumber)}`,
      undefined,
      options
    );
    return this.transformPhoneNumber(response);
  }

  /**
   * Update a phone number
   */
  async update(
    numberId: string,
    params: {
      friendlyName?: string;
      agentId?: string | null;
      config?: PhoneNumberConfig;
      metadata?: Metadata;
    },
    options?: RequestOptions
  ): Promise<PhoneNumber> {
    const response = await this.http.patch<PhoneNumber>(
      `/v1/phone-numbers/${numberId}`,
      {
        friendly_name: params.friendlyName,
        agent_id: params.agentId,
        config: params.config
          ? {
              voice_url: params.config.voiceUrl,
              voice_fallback_url: params.config.voiceFallbackUrl,
              status_callback_url: params.config.statusCallbackUrl,
              voice_method: params.config.voiceMethod,
              sms_url: params.config.smsUrl,
              sms_fallback_url: params.config.smsFallbackUrl,
              recording_enabled: params.config.recordingEnabled,
              transcription_enabled: params.config.transcriptionEnabled,
              caller_id_lookup: params.config.callerIdLookup,
            }
          : undefined,
        metadata: params.metadata,
      },
      options
    );
    return this.transformPhoneNumber(response);
  }

  /**
   * Release a phone number
   */
  async release(numberId: string, options?: RequestOptions): Promise<void> {
    await this.http.delete(`/v1/phone-numbers/${numberId}`, options);
  }

  // Number Purchase

  /**
   * Search for available phone numbers
   */
  async searchAvailable(
    request?: SearchNumbersRequest,
    options?: RequestOptions
  ): Promise<AvailableNumber[]> {
    const response = await this.http.get<{ available_numbers: AvailableNumber[] }>(
      '/v1/phone-numbers/available',
      {
        country_code: request?.countryCode || 'US',
        type: request?.type || 'local',
        area_code: request?.areaCode,
        contains: request?.contains,
        capabilities: request?.capabilities?.join(','),
        limit: request?.limit || 20,
      },
      options
    );
    return response.available_numbers.map((n) => this.transformAvailableNumber(n));
  }

  /**
   * Purchase an available phone number
   */
  async purchase(
    phoneNumber: string,
    params?: {
      friendlyName?: string;
      agentId?: string;
      config?: PhoneNumberConfig;
    },
    options?: RequestOptions
  ): Promise<PhoneNumber> {
    const response = await this.http.post<PhoneNumber>(
      '/v1/phone-numbers/purchase',
      {
        phone_number: phoneNumber,
        friendly_name: params?.friendlyName,
        agent_id: params?.agentId,
        config: params?.config
          ? {
              voice_url: params.config.voiceUrl,
              voice_fallback_url: params.config.voiceFallbackUrl,
              status_callback_url: params.config.statusCallbackUrl,
              voice_method: params.config.voiceMethod,
              sms_url: params.config.smsUrl,
              sms_fallback_url: params.config.smsFallbackUrl,
              recording_enabled: params.config.recordingEnabled,
              transcription_enabled: params.config.transcriptionEnabled,
              caller_id_lookup: params.config.callerIdLookup,
            }
          : undefined,
      },
      options
    );
    return this.transformPhoneNumber(response);
  }

  /**
   * Import an existing phone number from a provider
   */
  async importNumber(
    params: {
      phoneNumber: string;
      provider: string;
      providerSid: string;
      friendlyName?: string;
      agentId?: string;
    },
    options?: RequestOptions
  ): Promise<PhoneNumber> {
    const response = await this.http.post<PhoneNumber>(
      '/v1/phone-numbers/import',
      {
        phone_number: params.phoneNumber,
        provider: params.provider,
        provider_sid: params.providerSid,
        friendly_name: params.friendlyName,
        agent_id: params.agentId,
      },
      options
    );
    return this.transformPhoneNumber(response);
  }

  // Agent Assignment

  /**
   * Assign a phone number to an agent
   */
  async assignToAgent(
    numberId: string,
    agentId: string,
    options?: RequestOptions
  ): Promise<PhoneNumber> {
    return this.update(numberId, { agentId }, options);
  }

  /**
   * Unassign a phone number from its agent
   */
  async unassignFromAgent(numberId: string, options?: RequestOptions): Promise<PhoneNumber> {
    return this.update(numberId, { agentId: null }, options);
  }

  /**
   * Get all phone numbers assigned to an agent
   */
  async getAgentNumbers(agentId: string, options?: RequestOptions): Promise<PhoneNumber[]> {
    const response = await this.list({ agentId }, options);
    return response.items;
  }

  // Configuration

  /**
   * Get the configuration for a phone number
   */
  async getConfig(numberId: string, options?: RequestOptions): Promise<PhoneNumberConfig> {
    const response = await this.http.get<Record<string, unknown>>(
      `/v1/phone-numbers/${numberId}/config`,
      undefined,
      options
    );
    return this.transformConfig(response);
  }

  /**
   * Update the configuration for a phone number
   */
  async updateConfig(
    numberId: string,
    config: PhoneNumberConfig,
    options?: RequestOptions
  ): Promise<PhoneNumberConfig> {
    const response = await this.http.put<Record<string, unknown>>(
      `/v1/phone-numbers/${numberId}/config`,
      {
        voice_url: config.voiceUrl,
        voice_fallback_url: config.voiceFallbackUrl,
        status_callback_url: config.statusCallbackUrl,
        voice_method: config.voiceMethod,
        sms_url: config.smsUrl,
        sms_fallback_url: config.smsFallbackUrl,
        recording_enabled: config.recordingEnabled,
        transcription_enabled: config.transcriptionEnabled,
        caller_id_lookup: config.callerIdLookup,
      },
      options
    );
    return this.transformConfig(response);
  }

  // Verification

  /**
   * Start caller ID verification for outbound calls
   */
  async verifyCallerId(
    phoneNumber: string,
    friendlyName?: string,
    options?: RequestOptions
  ): Promise<{ verificationSid: string; status: string }> {
    const response = await this.http.post<{
      verification_sid: string;
      status: string;
    }>(
      '/v1/phone-numbers/verify',
      {
        phone_number: phoneNumber,
        friendly_name: friendlyName,
      },
      options
    );
    return {
      verificationSid: response.verification_sid,
      status: response.status,
    };
  }

  /**
   * Check the status of a caller ID verification
   */
  async checkVerification(
    verificationSid: string,
    options?: RequestOptions
  ): Promise<{ status: string; validationCode?: string }> {
    const response = await this.http.get<{
      status: string;
      validation_code?: string;
    }>(`/v1/phone-numbers/verify/${verificationSid}`, undefined, options);
    return {
      status: response.status,
      validationCode: response.validation_code,
    };
  }

  /**
   * Submit the verification code
   */
  async submitVerificationCode(
    verificationSid: string,
    code: string,
    options?: RequestOptions
  ): Promise<{ success: boolean; phoneNumber?: PhoneNumber }> {
    const response = await this.http.post<{
      success: boolean;
      phone_number?: PhoneNumber;
    }>(
      `/v1/phone-numbers/verify/${verificationSid}/submit`,
      { code },
      options
    );
    return {
      success: response.success,
      phoneNumber: response.phone_number
        ? this.transformPhoneNumber(response.phone_number)
        : undefined,
    };
  }

  // Convenience Methods

  /**
   * Get all active phone numbers
   */
  async getActiveNumbers(options?: RequestOptions): Promise<PhoneNumber[]> {
    const response = await this.list({ status: 'active' }, options);
    return response.items;
  }

  /**
   * Search for local numbers in an area code
   */
  async searchLocal(areaCode: string, limit = 20, options?: RequestOptions): Promise<AvailableNumber[]> {
    return this.searchAvailable(
      {
        type: 'local',
        areaCode,
        limit,
      },
      options
    );
  }

  /**
   * Search for toll-free numbers
   */
  async searchTollFree(
    contains?: string,
    limit = 20,
    options?: RequestOptions
  ): Promise<AvailableNumber[]> {
    return this.searchAvailable(
      {
        type: 'toll_free',
        contains,
        limit,
      },
      options
    );
  }

  /**
   * Purchase a number and assign it to an agent in one call
   */
  async purchaseAndAssign(
    phoneNumber: string,
    agentId: string,
    friendlyName?: string,
    options?: RequestOptions
  ): Promise<PhoneNumber> {
    return this.purchase(phoneNumber, { friendlyName, agentId }, options);
  }

  // Private helpers

  private transformPhoneNumber(data: Record<string, unknown>): PhoneNumber {
    return {
      id: data.id as string,
      number: data.number as string,
      friendlyName: (data.friendlyName || data.friendly_name) as string | undefined,
      type: data.type as PhoneNumberType,
      status: data.status as PhoneNumberStatus,
      capabilities: data.capabilities as PhoneNumberCapability[],
      agentId: (data.agentId || data.agent_id) as string | undefined,
      countryCode: (data.countryCode || data.country_code) as string,
      areaCode: (data.areaCode || data.area_code) as string | undefined,
      provider: data.provider as string,
      monthlyCost: (data.monthlyCost || data.monthly_cost) as number,
      createdAt: (data.createdAt || data.created_at) as string,
      updatedAt: (data.updatedAt || data.updated_at) as string,
      metadata: data.metadata as Metadata | undefined,
    };
  }

  private transformAvailableNumber(data: Record<string, unknown>): AvailableNumber {
    return {
      number: data.number as string,
      friendlyName: (data.friendlyName || data.friendly_name) as string,
      type: data.type as PhoneNumberType,
      capabilities: data.capabilities as PhoneNumberCapability[],
      countryCode: (data.countryCode || data.country_code) as string,
      areaCode: (data.areaCode || data.area_code) as string | undefined,
      region: data.region as string | undefined,
      locality: data.locality as string | undefined,
      monthlyCost: (data.monthlyCost || data.monthly_cost) as number,
      setupCost: (data.setupCost || data.setup_cost) as number,
    };
  }

  private transformConfig(data: Record<string, unknown>): PhoneNumberConfig {
    return {
      voiceUrl: (data.voiceUrl || data.voice_url) as string | undefined,
      voiceFallbackUrl: (data.voiceFallbackUrl || data.voice_fallback_url) as string | undefined,
      statusCallbackUrl: (data.statusCallbackUrl || data.status_callback_url) as string | undefined,
      voiceMethod: (data.voiceMethod || data.voice_method) as 'POST' | 'GET' | undefined,
      smsUrl: (data.smsUrl || data.sms_url) as string | undefined,
      smsFallbackUrl: (data.smsFallbackUrl || data.sms_fallback_url) as string | undefined,
      recordingEnabled: (data.recordingEnabled || data.recording_enabled) as boolean | undefined,
      transcriptionEnabled: (data.transcriptionEnabled || data.transcription_enabled) as
        | boolean
        | undefined,
      callerIdLookup: (data.callerIdLookup || data.caller_id_lookup) as boolean | undefined,
    };
  }
}

/**
 * Builder for phone number searches
 */
export class PhoneNumberSearchBuilder {
  private _countryCode = 'US';
  private _type: PhoneNumberType = 'local';
  private _areaCode?: string;
  private _contains?: string;
  private _capabilities: PhoneNumberCapability[] = [];
  private _limit = 20;

  country(code: string): this {
    this._countryCode = code;
    return this;
  }

  type(numberType: PhoneNumberType): this {
    this._type = numberType;
    return this;
  }

  local(): this {
    this._type = 'local';
    return this;
  }

  tollFree(): this {
    this._type = 'toll_free';
    return this;
  }

  mobile(): this {
    this._type = 'mobile';
    return this;
  }

  areaCode(code: string): this {
    this._areaCode = code;
    return this;
  }

  contains(pattern: string): this {
    this._contains = pattern;
    return this;
  }

  withVoice(): this {
    if (!this._capabilities.includes('voice')) {
      this._capabilities.push('voice');
    }
    return this;
  }

  withSms(): this {
    if (!this._capabilities.includes('sms')) {
      this._capabilities.push('sms');
    }
    return this;
  }

  withMms(): this {
    if (!this._capabilities.includes('mms')) {
      this._capabilities.push('mms');
    }
    return this;
  }

  limit(count: number): this {
    this._limit = count;
    return this;
  }

  build(): SearchNumbersRequest {
    return {
      countryCode: this._countryCode,
      type: this._type,
      areaCode: this._areaCode,
      contains: this._contains,
      capabilities: this._capabilities.length > 0 ? this._capabilities : undefined,
      limit: this._limit,
    };
  }

  async search(api: PhoneNumbersAPI): Promise<AvailableNumber[]> {
    return api.searchAvailable(this.build());
  }
}
