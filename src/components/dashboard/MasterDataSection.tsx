import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { CustomSelect, SelectOption } from './CustomSelect';
import { ChevronDown, ChevronRight, Eye, Edit, Plus } from 'lucide-react';
import { cn } from '@/lib/utils';

interface MasterDataRow {
  id: string;
  projectName: string;
  reportNeeded: string;
  filedWithRBI: string;
  annualReportFilingDate: string;
  complianceTracker: string;
  actions: string;
}

const masterData: MasterDataRow[] = [
  {
    id: 'VIZ200004FORTO002047',
    projectName: 'Foreign Investor Profile Upload',
    reportNeeded: '',
    filedWithRBI: '',
    annualReportFilingDate: '',
    complianceTracker: 'Allocated',
    actions: ''
  },
  {
    id: 'VIZ200004FORTO002048', 
    projectName: 'Foreign Investor Profile Upload',
    reportNeeded: '',
    filedWithRBI: '',
    annualReportFilingDate: '',
    complianceTracker: 'Allocated',
    actions: ''
  }
];

const selectOptions: SelectOption[] = [
  { value: 'yes', label: 'Yes' },
  { value: 'no', label: 'No' },
  { value: 'pending', label: 'Pending' },
  { value: 'completed', label: 'Completed' }
];

const multiSelectOptions: SelectOption[] = [
  { value: 'option1', label: 'Option 1' },
  { value: 'option2', label: 'Option 2' },
  { value: 'option3', label: 'Option 3' },
  { value: 'option4', label: 'Option 4' },
  { value: 'option5', label: 'Option 5' }
];

export const MasterDataSection: React.FC = () => {
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [singleSelectValue, setSingleSelectValue] = useState<string>('');
  const [multiSelectValue, setMultiSelectValue] = useState<string[]>([]);
  const [searchableMultiSelectValue, setSearchableMultiSelectValue] = useState<string[]>([]);

  const toggleRowExpansion = (id: string) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(id)) {
      newExpanded.delete(id);
    } else {
      newExpanded.add(id);
    }
    setExpandedRows(newExpanded);
  };

  return (
    <div className="space-y-6">
      {/* Custom Select Components Demo */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Custom Select Components</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Single Select</label>
              <CustomSelect
                options={selectOptions}
                value={singleSelectValue}
                onChange={(value) => setSingleSelectValue(value as string)}
                placeholder="Select an option..."
              />
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Multi Select (No Search)</label>
              <CustomSelect
                options={multiSelectOptions}
                value={multiSelectValue}
                onChange={(value) => setMultiSelectValue(value as string[])}
                placeholder="Select multiple..."
                multiple={true}
                searchable={false}
              />
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Multi Select with Search</label>
              <CustomSelect
                options={multiSelectOptions}
                value={searchableMultiSelectValue}
                onChange={(value) => setSearchableMultiSelectValue(value as string[])}
                placeholder="Search and select..."
                multiple={true}
                searchable={true}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Master Data Table */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-lg">Master Data</CardTitle>
          <Button size="sm">
            <Plus className="h-4 w-4 mr-2" />
            Add Entry
          </Button>
        </CardHeader>
        <CardContent>
          <div className="rounded-lg border border-table-border bg-card">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-table-header text-table-header-foreground">
                    <th className="px-4 py-3 text-left text-sm font-medium w-8"></th>
                    <th className="px-4 py-3 text-left text-sm font-medium">Projects</th>
                    <th className="px-4 py-3 text-left text-sm font-medium">Report Needed</th>
                    <th className="px-4 py-3 text-left text-sm font-medium">Filed With RBI</th>
                    <th className="px-4 py-3 text-left text-sm font-medium">Annual Report Filing Date</th>
                    <th className="px-4 py-3 text-left text-sm font-medium">Compliance Tracker</th>
                    <th className="px-4 py-3 text-center text-sm font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {masterData.map((row, index) => (
                    <React.Fragment key={row.id}>
                      <tr className="border-t border-table-border bg-table-row hover:bg-table-row-hover transition-colors">
                        <td className="px-4 py-3">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0"
                            onClick={() => toggleRowExpansion(row.id)}
                          >
                            {expandedRows.has(row.id) ? (
                              <ChevronDown className="h-4 w-4" />
                            ) : (
                              <ChevronRight className="h-4 w-4" />
                            )}
                          </Button>
                        </td>
                        <td className="px-4 py-3 text-sm">
                          <div>
                            <p className="font-medium text-foreground">{row.projectName}</p>
                            <p className="text-xs text-muted-foreground">{row.id}</p>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-sm text-muted-foreground">
                          {row.reportNeeded || '-'}
                        </td>
                        <td className="px-4 py-3 text-sm text-muted-foreground">
                          {row.filedWithRBI || '-'}
                        </td>
                        <td className="px-4 py-3 text-sm text-muted-foreground">
                          {row.annualReportFilingDate || '-'}
                        </td>
                        <td className="px-4 py-3">
                          <Badge variant="secondary">
                            {row.complianceTracker}
                          </Badge>
                        </td>
                        <td className="px-4 py-3 text-center">
                          <div className="flex items-center justify-center gap-1">
                            <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                              <Eye className="h-4 w-4" />
                            </Button>
                            <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                              <Edit className="h-4 w-4" />
                            </Button>
                          </div>
                        </td>
                      </tr>
                      
                      {expandedRows.has(row.id) && (
                        <tr className="border-t border-table-border bg-muted/50">
                          <td colSpan={7} className="px-4 py-4">
                            <div className="space-y-3">
                              <h4 className="text-sm font-medium text-foreground">Additional Details</h4>
                              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
                                <div>
                                  <span className="text-muted-foreground">Status:</span>
                                  <span className="ml-2 text-foreground">Active</span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">Created:</span>
                                  <span className="ml-2 text-foreground">2024-01-15</span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">Last Updated:</span>
                                  <span className="ml-2 text-foreground">2024-02-20</span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">Assigned To:</span>
                                  <span className="ml-2 text-foreground">John Doe</span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">Priority:</span>
                                  <Badge variant="outline" className="ml-2">Medium</Badge>
                                </div>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};