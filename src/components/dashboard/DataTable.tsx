import React from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { MoreHorizontal, Edit, Trash2 } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

interface DataRow {
  id: number;
  applicationForm: string;
  issueDate: string;
  onlineSubmission: string;
  applicationFormFilled: string;
  status: string;
}

interface DataTableProps {
  data: DataRow[];
}

export const DataTable: React.FC<DataTableProps> = ({ data }) => {
  const getStatusBadgeVariant = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'default';
      case 'pending':
        return 'secondary';
      default:
        return 'outline';
    }
  };

  const getYesNoBadge = (value: string) => {
    return value.toLowerCase() === 'yes' ? (
      <Badge variant="default" className="bg-success hover:bg-success/80">Yes</Badge>
    ) : (
      <Badge variant="outline">No</Badge>
    );
  };

  return (
    <div className="rounded-lg border border-table-border bg-card">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-table-header text-table-header-foreground">
              <th className="px-4 py-3 text-left text-sm font-medium">Application Form Name</th>
              <th className="px-4 py-3 text-left text-sm font-medium">Issue Date</th>
              <th className="px-4 py-3 text-left text-sm font-medium">Whether online confirmation</th>
              <th className="px-4 py-3 text-left text-sm font-medium">Whether application form filled</th>
              <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
              <th className="px-4 py-3 text-center text-sm font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, index) => (
              <tr 
                key={row.id}
                className="border-t border-table-border bg-table-row hover:bg-table-row-hover transition-colors"
              >
                <td className="px-4 py-3 text-sm text-foreground font-medium">
                  {row.applicationForm}
                </td>
                <td className="px-4 py-3 text-sm text-muted-foreground">
                  {row.issueDate}
                </td>
                <td className="px-4 py-3">
                  {getYesNoBadge(row.onlineSubmission)}
                </td>
                <td className="px-4 py-3">
                  {getYesNoBadge(row.applicationFormFilled)}
                </td>
                <td className="px-4 py-3">
                  <Badge variant={getStatusBadgeVariant(row.status)}>
                    {row.status}
                  </Badge>
                </td>
                <td className="px-4 py-3 text-center">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-48">
                      <DropdownMenuItem className="cursor-pointer">
                        <Edit className="h-4 w-4 mr-2" />
                        Edit
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem className="cursor-pointer text-destructive focus:text-destructive">
                        <Trash2 className="h-4 w-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};