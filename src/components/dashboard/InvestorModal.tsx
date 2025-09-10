import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Download, Upload, X } from 'lucide-react';

interface InvestorModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const investorData = [
  {
    id: 1,
    name: 'Ramanathan',
    address: 'Address',
    contribution: 'Contractual',
    residentialStatus: 'Resident',
    reportNeeded: 'No',
    amount: '₹10,00,000',
    percentage: '50%',
    actions: 'Active'
  },
  {
    id: 2,
    name: 'Dr Praka Anand',
    address: 'Adyar, Chennai',
    contribution: 'Economic Club',
    residentialStatus: 'Non Resident',
    reportNeeded: 'Non Resident',
    amount: '₹10,000',
    percentage: '50%',
    actions: 'Active'
  },
  {
    id: 3,
    name: 'Dr Antosh Kappor',
    address: 'Adyar, Chennai',
    contribution: 'Economic Club',
    residentialStatus: 'Non Resident',
    reportNeeded: 'Non Resident',
    amount: '₹10,000',
    percentage: '50%',
    actions: 'Active'
  }
];

export const InvestorModal: React.FC<InvestorModalProps> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState('general');

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-hidden">
        <DialogHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
          <DialogTitle className="text-xl font-semibold">Investor Holding</DialogTitle>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Download Template
            </Button>
            <Button variant="outline" size="sm">
              <Upload className="h-4 w-4 mr-2" />
              Upload Excel
            </Button>
            <Button variant="outline" size="sm" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="general">General</TabsTrigger>
            <TabsTrigger value="shares">Shares</TabsTrigger>
            <TabsTrigger value="debt">Debt</TabsTrigger>
            <TabsTrigger value="others">Others</TabsTrigger>
          </TabsList>

          <TabsContent value="general" className="mt-4 space-y-4">
            <div className="rounded-lg border border-table-border bg-card">
              <div className="overflow-x-auto max-h-96">
                <table className="w-full">
                  <thead className="sticky top-0">
                    <tr className="bg-table-header text-table-header-foreground">
                      <th className="px-4 py-3 text-left text-sm font-medium">Name of Investors</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Address</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Contribution</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Residential Status</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Report Needed</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Amount</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Percentage</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {investorData.map((investor, index) => (
                      <tr 
                        key={investor.id}
                        className="border-t border-table-border bg-table-row hover:bg-table-row-hover transition-colors"
                      >
                        <td className="px-4 py-3 text-sm font-medium text-foreground">
                          {investor.name}
                        </td>
                        <td className="px-4 py-3 text-sm text-muted-foreground">
                          {investor.address}
                        </td>
                        <td className="px-4 py-3 text-sm text-muted-foreground">
                          {investor.contribution}
                        </td>
                        <td className="px-4 py-3 text-sm">
                          <Badge variant={investor.residentialStatus === 'Resident' ? 'default' : 'secondary'}>
                            {investor.residentialStatus}
                          </Badge>
                        </td>
                        <td className="px-4 py-3 text-sm">
                          <Badge variant={investor.reportNeeded === 'No' ? 'outline' : 'secondary'}>
                            {investor.reportNeeded}
                          </Badge>
                        </td>
                        <td className="px-4 py-3 text-sm font-medium text-foreground">
                          {investor.amount}
                        </td>
                        <td className="px-4 py-3 text-sm text-muted-foreground">
                          {investor.percentage}
                        </td>
                        <td className="px-4 py-3 text-sm">
                          <Badge variant="default" className="bg-success hover:bg-success/80">
                            {investor.actions}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="shares" className="mt-4">
            <div className="flex items-center justify-center h-64 text-muted-foreground">
              <p>Shares data will be displayed here</p>
            </div>
          </TabsContent>

          <TabsContent value="debt" className="mt-4">
            <div className="flex items-center justify-center h-64 text-muted-foreground">
              <p>Debt information will be displayed here</p>
            </div>
          </TabsContent>

          <TabsContent value="others" className="mt-4">
            <div className="flex items-center justify-center h-64 text-muted-foreground">
              <p>Other details will be displayed here</p>
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
};