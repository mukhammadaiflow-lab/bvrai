import type { BuilderEngine } from '../client';
import type { Subscription, Invoice, PaymentMethod, Usage, PaginatedResponse, ListParams } from '../models';

export class BillingResource {
  constructor(private readonly client: BuilderEngine) {}

  async getSubscription(): Promise<Subscription> {
    return this.client.request({ method: 'GET', path: '/api/v1/billing/subscription' });
  }

  async getUsage(period: string = 'current'): Promise<Usage> {
    return this.client.request({ method: 'GET', path: '/api/v1/billing/usage', params: { period } });
  }

  async listInvoices(params: ListParams = {}): Promise<PaginatedResponse<Invoice>> {
    return this.client.request({ method: 'GET', path: '/api/v1/billing/invoices', params });
  }

  async listPaymentMethods(): Promise<PaymentMethod[]> {
    const response = await this.client.request<{ payment_methods: PaymentMethod[] }>({ method: 'GET', path: '/api/v1/billing/payment-methods' });
    return response.payment_methods;
  }

  async createCheckoutSession(planId: string, successUrl: string, cancelUrl: string): Promise<{ url: string }> {
    return this.client.request({ method: 'POST', path: '/api/v1/billing/checkout', body: { plan_id: planId, success_url: successUrl, cancel_url: cancelUrl } });
  }
}
