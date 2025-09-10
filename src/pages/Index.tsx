import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { DataTable } from '@/components/dashboard/DataTable';
import { InvestorModal } from '@/components/dashboard/InvestorModal';
import { FileUploadModal } from '@/components/dashboard/FileUploadModal';
import { MasterDataSection } from '@/components/dashboard/MasterDataSection';
import { NotificationBanner } from '@/components/dashboard/NotificationBanner';
import { TaxFormModal } from '@/components/forms/TaxFormModal';
import { InvestorFormModal } from '@/components/forms/InvestorFormModal';
import { Plus, Download, Upload, FileText, Users, FileInput } from 'lucide-react';

const Index = () => {
  const [showInvestorModal, setShowInvestorModal] = useState(false);
  const [showFileUploadModal, setShowFileUploadModal] = useState(false);
  const [showTaxFormModal, setShowTaxFormModal] = useState(false);
  const [showInvestorFormModal, setShowInvestorFormModal] = useState(false);

  const sampleData = [
    {
      id: 1,
      applicationForm: 'Aditiya Kumar',
      issueDate: '31/07/24',
      onlineSubmission: 'No',
      applicationFormFilled: 'Yes',
      status: 'Pending'
    },
    {
      id: 2,
      applicationForm: 'Aditiya Kumar',
      issueDate: '31/07/24',
      onlineSubmission: 'No',
      applicationFormFilled: 'Yes',
      status: 'Completed'
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-semibold text-foreground">Tax & RBI Dashboard</h1>
              <p className="text-sm text-muted-foreground">Generate and manage tax forms and RBI reports</p>
            </div>
            <div className="flex gap-3">
              <Button onClick={() => setShowFileUploadModal(true)} variant="outline">
                <Upload className="h-4 w-4 mr-2" />
                Upload Files
              </Button>
              <Button onClick={() => setShowInvestorFormModal(true)} variant="outline">
                <Users className="h-4 w-4 mr-2" />
                Add Investors
              </Button>
              <Button onClick={() => setShowTaxFormModal(true)}>
                <FileInput className="h-4 w-4 mr-2" />
                New Tax Form
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-6 space-y-6">
        {/* Notification Banner */}
        <NotificationBanner />

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Forms */}
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-lg">Foreign Exchange Management Applications</CardTitle>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => setShowInvestorModal(true)}
                >
                  <FileText className="h-4 w-4 mr-2" />
                  View Details
                </Button>
              </CardHeader>
              <CardContent>
                <DataTable data={sampleData} />
              </CardContent>
            </Card>

            <MasterDataSection />
          </div>

          {/* Right Column - Statistics */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Quick Stats</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-muted rounded-lg">
                  <span className="text-sm font-medium">Pending Forms</span>
                  <span className="text-lg font-semibold text-warning">12</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-muted rounded-lg">
                  <span className="text-sm font-medium">Completed Forms</span>
                  <span className="text-lg font-semibold text-success">48</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-muted rounded-lg">
                  <span className="text-sm font-medium">Generated Reports</span>
                  <span className="text-lg font-semibold text-primary">156</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Recent Downloads</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-3 p-2 hover:bg-muted rounded-lg cursor-pointer">
                  <FileText className="h-4 w-4 text-primary" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">Form-A Report.pdf</p>
                    <p className="text-xs text-muted-foreground">2 hours ago</p>
                  </div>
                  <Download className="h-4 w-4 text-muted-foreground" />
                </div>
                <div className="flex items-center gap-3 p-2 hover:bg-muted rounded-lg cursor-pointer">
                  <FileText className="h-4 w-4 text-primary" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">RBI-Application.docx</p>
                    <p className="text-xs text-muted-foreground">1 day ago</p>
                  </div>
                  <Download className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      {/* Modals */}
      <InvestorModal 
        isOpen={showInvestorModal} 
        onClose={() => setShowInvestorModal(false)} 
      />
      <FileUploadModal 
        isOpen={showFileUploadModal} 
        onClose={() => setShowFileUploadModal(false)} 
      />
      <TaxFormModal 
        isOpen={showTaxFormModal} 
        onClose={() => setShowTaxFormModal(false)} 
      />
      <InvestorFormModal 
        isOpen={showInvestorFormModal} 
        onClose={() => setShowInvestorFormModal(false)} 
      />
    </div>
  );
};

export default Index;