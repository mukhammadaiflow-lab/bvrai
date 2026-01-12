"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Phone,
  Plus,
  Search,
  MoreVertical,
  Globe,
  Bot,
  Trash2,
  Settings,
  CheckCircle,
  XCircle,
  Copy,
  ExternalLink,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { phoneNumbersApi, agentsApi } from "@/lib/api";
import { formatRelativeTime, cn } from "@/lib/utils";

interface PhoneNumber {
  id: string;
  phone_number: string;
  country_code: string;
  region: string;
  capabilities: string[];
  status: "active" | "inactive" | "pending";
  provider: string;
  monthly_cost: number;
  assigned_agent_id?: string;
  assigned_agent_name?: string;
  created_at: string;
  last_used_at?: string;
}

interface AvailableNumber {
  phone_number: string;
  country_code: string;
  region: string;
  capabilities: string[];
  monthly_cost: number;
  setup_cost: number;
}

// Phone Number Row Component
function PhoneNumberRow({
  number,
  onAssign,
  onRelease,
  onConfigure,
}: {
  number: PhoneNumber;
  onAssign: (id: string) => void;
  onRelease: (id: string) => void;
  onConfigure: (id: string) => void;
}) {
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <TableRow>
      <TableCell>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
            <Phone className="h-5 w-5 text-primary" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-medium">{number.phone_number}</span>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={() => copyToClipboard(number.phone_number)}
              >
                <Copy className="h-3 w-3" />
              </Button>
            </div>
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <Globe className="h-3 w-3" />
              {number.region}, {number.country_code}
            </div>
          </div>
        </div>
      </TableCell>
      <TableCell>
        <div className="flex flex-wrap gap-1">
          {number.capabilities.map((cap) => (
            <Badge key={cap} variant="outline" className="text-xs">
              {cap}
            </Badge>
          ))}
        </div>
      </TableCell>
      <TableCell>
        <Badge
          variant={
            number.status === "active"
              ? "default"
              : number.status === "pending"
              ? "secondary"
              : "outline"
          }
          className={cn(
            number.status === "active" && "bg-success text-success-foreground"
          )}
        >
          {number.status === "active" && <CheckCircle className="mr-1 h-3 w-3" />}
          {number.status === "inactive" && <XCircle className="mr-1 h-3 w-3" />}
          {number.status}
        </Badge>
      </TableCell>
      <TableCell>
        {number.assigned_agent_id ? (
          <div className="flex items-center gap-2">
            <Bot className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm">{number.assigned_agent_name}</span>
          </div>
        ) : (
          <span className="text-sm text-muted-foreground">Not assigned</span>
        )}
      </TableCell>
      <TableCell>
        <span className="text-sm">${number.monthly_cost.toFixed(2)}/mo</span>
      </TableCell>
      <TableCell>
        <span className="text-sm text-muted-foreground">
          {number.last_used_at ? formatRelativeTime(number.last_used_at) : "Never"}
        </span>
      </TableCell>
      <TableCell>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon">
              <MoreVertical className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => onConfigure(number.id)}>
              <Settings className="mr-2 h-4 w-4" />
              Configure
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onAssign(number.id)}>
              <Bot className="mr-2 h-4 w-4" />
              {number.assigned_agent_id ? "Reassign Agent" : "Assign Agent"}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={() => onRelease(number.id)}
              className="text-destructive focus:text-destructive"
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Release Number
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </TableCell>
    </TableRow>
  );
}

// Buy Number Dialog
function BuyNumberDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [step, setStep] = useState<"search" | "select" | "confirm">("search");
  const [searchParams, setSearchParams] = useState({
    country: "US",
    region: "",
    areaCode: "",
    capabilities: [] as string[],
  });
  const [selectedNumber, setSelectedNumber] = useState<AvailableNumber | null>(null);
  const queryClient = useQueryClient();

  // Search for available numbers
  const { data: availableNumbers, isLoading: isSearching } = useQuery({
    queryKey: ["available-numbers", searchParams],
    queryFn: () => phoneNumbersApi.searchAvailable(searchParams),
    enabled: step === "select",
  });

  // Purchase mutation
  const purchaseMutation = useMutation({
    mutationFn: (phoneNumber: string) => phoneNumbersApi.purchase(phoneNumber),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["phone-numbers"] });
      onOpenChange(false);
      setStep("search");
      setSelectedNumber(null);
    },
  });

  const mockAvailableNumbers: AvailableNumber[] = [
    {
      phone_number: "+1 (415) 555-0101",
      country_code: "US",
      region: "California",
      capabilities: ["voice", "sms"],
      monthly_cost: 1.5,
      setup_cost: 0,
    },
    {
      phone_number: "+1 (415) 555-0102",
      country_code: "US",
      region: "California",
      capabilities: ["voice", "sms", "mms"],
      monthly_cost: 2.0,
      setup_cost: 0,
    },
    {
      phone_number: "+1 (415) 555-0103",
      country_code: "US",
      region: "California",
      capabilities: ["voice"],
      monthly_cost: 1.0,
      setup_cost: 0,
    },
    {
      phone_number: "+1 (650) 555-0201",
      country_code: "US",
      region: "California",
      capabilities: ["voice", "sms"],
      monthly_cost: 1.5,
      setup_cost: 0,
    },
    {
      phone_number: "+1 (510) 555-0301",
      country_code: "US",
      region: "California",
      capabilities: ["voice", "sms", "mms"],
      monthly_cost: 2.0,
      setup_cost: 0,
    },
  ];

  const countries = [
    { code: "US", name: "United States" },
    { code: "CA", name: "Canada" },
    { code: "GB", name: "United Kingdom" },
    { code: "AU", name: "Australia" },
    { code: "DE", name: "Germany" },
  ];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {step === "search" && "Get a Phone Number"}
            {step === "select" && "Select a Number"}
            {step === "confirm" && "Confirm Purchase"}
          </DialogTitle>
          <DialogDescription>
            {step === "search" && "Search for available phone numbers in your desired region"}
            {step === "select" && "Choose a number that fits your needs"}
            {step === "confirm" && "Review and confirm your purchase"}
          </DialogDescription>
        </DialogHeader>

        {step === "search" && (
          <div className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label>Country</Label>
                <Select
                  value={searchParams.country}
                  onValueChange={(value) => setSearchParams({ ...searchParams, country: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {countries.map((country) => (
                      <SelectItem key={country.code} value={country.code}>
                        {country.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Area Code (optional)</Label>
                <Input
                  placeholder="e.g., 415"
                  value={searchParams.areaCode}
                  onChange={(e) => setSearchParams({ ...searchParams, areaCode: e.target.value })}
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Capabilities</Label>
              <div className="flex flex-wrap gap-2">
                {["voice", "sms", "mms"].map((cap) => (
                  <Button
                    key={cap}
                    variant={searchParams.capabilities.includes(cap) ? "default" : "outline"}
                    size="sm"
                    onClick={() => {
                      setSearchParams({
                        ...searchParams,
                        capabilities: searchParams.capabilities.includes(cap)
                          ? searchParams.capabilities.filter((c) => c !== cap)
                          : [...searchParams.capabilities, cap],
                      });
                    }}
                  >
                    {cap.toUpperCase()}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        )}

        {step === "select" && (
          <div className="space-y-4">
            {isSearching ? (
              <div className="flex items-center justify-center py-8">
                <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
              </div>
            ) : (
              <div className="max-h-[400px] overflow-y-auto">
                <div className="space-y-2">
                  {(availableNumbers?.numbers || mockAvailableNumbers).map((num) => (
                    <div
                      key={num.phone_number}
                      className={cn(
                        "flex items-center justify-between rounded-lg border p-4 cursor-pointer transition-colors",
                        selectedNumber?.phone_number === num.phone_number
                          ? "border-primary bg-primary/5"
                          : "hover:border-primary/50"
                      )}
                      onClick={() => setSelectedNumber(num)}
                    >
                      <div className="flex items-center gap-3">
                        <Phone className="h-5 w-5 text-muted-foreground" />
                        <div>
                          <p className="font-medium">{num.phone_number}</p>
                          <p className="text-sm text-muted-foreground">{num.region}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="flex gap-1">
                          {num.capabilities.map((cap) => (
                            <Badge key={cap} variant="outline" className="text-xs">
                              {cap}
                            </Badge>
                          ))}
                        </div>
                        <span className="text-sm font-medium">
                          ${num.monthly_cost.toFixed(2)}/mo
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {step === "confirm" && selectedNumber && (
          <div className="space-y-4">
            <div className="rounded-lg border p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Phone className="h-8 w-8 text-primary" />
                  <div>
                    <p className="text-xl font-bold">{selectedNumber.phone_number}</p>
                    <p className="text-sm text-muted-foreground">
                      {selectedNumber.region}, {selectedNumber.country_code}
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between py-2">
                <span className="text-muted-foreground">Capabilities</span>
                <span>{selectedNumber.capabilities.join(", ").toUpperCase()}</span>
              </div>
              <div className="flex justify-between py-2 border-t">
                <span className="text-muted-foreground">Setup Fee</span>
                <span>${selectedNumber.setup_cost.toFixed(2)}</span>
              </div>
              <div className="flex justify-between py-2 border-t">
                <span className="text-muted-foreground">Monthly Cost</span>
                <span>${selectedNumber.monthly_cost.toFixed(2)}/mo</span>
              </div>
              <div className="flex justify-between py-2 border-t font-medium">
                <span>Total Due Today</span>
                <span>
                  ${(selectedNumber.setup_cost + selectedNumber.monthly_cost).toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        )}

        <DialogFooter>
          {step === "search" && (
            <Button onClick={() => setStep("select")}>
              <Search className="mr-2 h-4 w-4" />
              Search Numbers
            </Button>
          )}
          {step === "select" && (
            <>
              <Button variant="outline" onClick={() => setStep("search")}>
                Back
              </Button>
              <Button onClick={() => setStep("confirm")} disabled={!selectedNumber}>
                Continue
              </Button>
            </>
          )}
          {step === "confirm" && (
            <>
              <Button variant="outline" onClick={() => setStep("select")}>
                Back
              </Button>
              <Button
                onClick={() => purchaseMutation.mutate(selectedNumber!.phone_number)}
                disabled={purchaseMutation.isPending}
              >
                {purchaseMutation.isPending ? "Purchasing..." : "Confirm Purchase"}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// Assign Agent Dialog
function AssignAgentDialog({
  open,
  onOpenChange,
  phoneNumberId,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  phoneNumberId: string | null;
}) {
  const [selectedAgentId, setSelectedAgentId] = useState<string>("");
  const queryClient = useQueryClient();

  const { data: agentsData } = useQuery({
    queryKey: ["agents-list"],
    queryFn: () => agentsApi.list({ page_size: 100 }),
  });

  const assignMutation = useMutation({
    mutationFn: ({ numberId, agentId }: { numberId: string; agentId: string }) =>
      phoneNumbersApi.assignAgent(numberId, agentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["phone-numbers"] });
      onOpenChange(false);
    },
  });

  const agents = agentsData?.agents || [];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Assign Agent</DialogTitle>
          <DialogDescription>
            Select an agent to handle calls on this number
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Select Agent</Label>
            <Select value={selectedAgentId} onValueChange={setSelectedAgentId}>
              <SelectTrigger>
                <SelectValue placeholder="Choose an agent" />
              </SelectTrigger>
              <SelectContent>
                {agents.map((agent: { id: string; name: string }) => (
                  <SelectItem key={agent.id} value={agent.id}>
                    {agent.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={() =>
              assignMutation.mutate({ numberId: phoneNumberId!, agentId: selectedAgentId })
            }
            disabled={!selectedAgentId || assignMutation.isPending}
          >
            {assignMutation.isPending ? "Assigning..." : "Assign Agent"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default function PhoneNumbersPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [buyDialogOpen, setBuyDialogOpen] = useState(false);
  const [assignDialogOpen, setAssignDialogOpen] = useState(false);
  const [selectedNumberId, setSelectedNumberId] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Fetch phone numbers
  const { data: numbersData, isLoading } = useQuery({
    queryKey: ["phone-numbers"],
    queryFn: () => phoneNumbersApi.list(),
  });

  // Release mutation
  const releaseMutation = useMutation({
    mutationFn: (id: string) => phoneNumbersApi.release(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["phone-numbers"] });
    },
  });

  // Mock data
  const mockNumbers: PhoneNumber[] = [
    {
      id: "1",
      phone_number: "+1 (415) 555-0123",
      country_code: "US",
      region: "California",
      capabilities: ["voice", "sms"],
      status: "active",
      provider: "twilio",
      monthly_cost: 1.5,
      assigned_agent_id: "agent-1",
      assigned_agent_name: "Sales Assistant",
      created_at: "2024-01-15T10:00:00Z",
      last_used_at: "2024-01-20T14:30:00Z",
    },
    {
      id: "2",
      phone_number: "+1 (650) 555-0456",
      country_code: "US",
      region: "California",
      capabilities: ["voice", "sms", "mms"],
      status: "active",
      provider: "twilio",
      monthly_cost: 2.0,
      assigned_agent_id: "agent-2",
      assigned_agent_name: "Support Bot",
      created_at: "2024-01-10T09:00:00Z",
      last_used_at: "2024-01-20T16:45:00Z",
    },
    {
      id: "3",
      phone_number: "+1 (510) 555-0789",
      country_code: "US",
      region: "California",
      capabilities: ["voice"],
      status: "inactive",
      provider: "twilio",
      monthly_cost: 1.0,
      created_at: "2024-01-05T08:00:00Z",
    },
    {
      id: "4",
      phone_number: "+44 20 7123 4567",
      country_code: "GB",
      region: "London",
      capabilities: ["voice", "sms"],
      status: "active",
      provider: "twilio",
      monthly_cost: 2.5,
      assigned_agent_id: "agent-3",
      assigned_agent_name: "UK Support",
      created_at: "2024-01-01T12:00:00Z",
      last_used_at: "2024-01-19T11:20:00Z",
    },
  ];

  const numbers = numbersData?.phone_numbers || mockNumbers;

  // Filter numbers
  const filteredNumbers = numbers.filter((num: PhoneNumber) =>
    num.phone_number.toLowerCase().includes(searchQuery.toLowerCase()) ||
    num.region.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Stats
  const activeCount = numbers.filter((n: PhoneNumber) => n.status === "active").length;
  const totalMonthlyCost = numbers.reduce((sum: number, n: PhoneNumber) => sum + n.monthly_cost, 0);

  const handleAssign = (id: string) => {
    setSelectedNumberId(id);
    setAssignDialogOpen(true);
  };

  const handleRelease = (id: string) => {
    if (confirm("Are you sure you want to release this number? This cannot be undone.")) {
      releaseMutation.mutate(id);
    }
  };

  const handleConfigure = (id: string) => {
    console.log("Configure number:", id);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold">Phone Numbers</h1>
          <p className="text-muted-foreground">
            Manage your phone numbers and agent assignments
          </p>
        </div>
        <Button onClick={() => setBuyDialogOpen(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Get Number
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Total Numbers</CardTitle>
            <Phone className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{numbers.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Active</CardTitle>
            <CheckCircle className="h-4 w-4 text-success" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{activeCount}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Assigned to Agents</CardTitle>
            <Bot className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {numbers.filter((n: PhoneNumber) => n.assigned_agent_id).length}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Monthly Cost</CardTitle>
            <span className="text-sm font-medium text-muted-foreground">USD</span>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">${totalMonthlyCost.toFixed(2)}</div>
          </CardContent>
        </Card>
      </div>

      {/* Numbers Table */}
      <Card>
        <CardHeader>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle>Your Numbers</CardTitle>
              <CardDescription>All phone numbers in your account</CardDescription>
            </div>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search numbers..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 w-[300px]"
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
            </div>
          ) : filteredNumbers.length === 0 ? (
            <div className="text-center py-8">
              <Phone className="mx-auto h-12 w-12 text-muted-foreground/50" />
              <h3 className="mt-4 text-lg font-medium">No phone numbers found</h3>
              <p className="mt-2 text-sm text-muted-foreground">
                {searchQuery
                  ? "Try adjusting your search"
                  : "Get your first phone number to start receiving calls"}
              </p>
              {!searchQuery && (
                <Button className="mt-4" onClick={() => setBuyDialogOpen(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  Get Number
                </Button>
              )}
            </div>
          ) : (
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Number</TableHead>
                    <TableHead>Capabilities</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Assigned Agent</TableHead>
                    <TableHead>Cost</TableHead>
                    <TableHead>Last Used</TableHead>
                    <TableHead className="w-[50px]"></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredNumbers.map((number: PhoneNumber) => (
                    <PhoneNumberRow
                      key={number.id}
                      number={number}
                      onAssign={handleAssign}
                      onRelease={handleRelease}
                      onConfigure={handleConfigure}
                    />
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Dialogs */}
      <BuyNumberDialog open={buyDialogOpen} onOpenChange={setBuyDialogOpen} />
      <AssignAgentDialog
        open={assignDialogOpen}
        onOpenChange={setAssignDialogOpen}
        phoneNumberId={selectedNumberId}
      />
    </div>
  );
}
