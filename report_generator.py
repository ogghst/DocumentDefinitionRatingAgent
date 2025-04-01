import os
import json
# import pandas as pd # Not used
import matplotlib.pyplot as plt
import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from io import BytesIO
from datetime import datetime
# import matplotlib.patches as mpatches # Not used

# Set the matplotlib backend to Agg for saving figures
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid') # Use a clean seaborn style

# --- Minimalist Color Palette ---
COLOR_PRIMARY = colors.HexColor('#007AFF') # Blue
COLOR_SUCCESS = colors.HexColor('#34C759') # Green
COLOR_WARNING = colors.HexColor('#FF9500') # Orange
COLOR_ERROR = colors.HexColor('#FF3B30') # Red
COLOR_TEXT_PRIMARY = colors.HexColor('#1C1C1E') # Almost Black
COLOR_TEXT_SECONDARY = colors.HexColor('#8E8E93') # Gray
COLOR_BACKGROUND_LIGHT = colors.HexColor('#F2F2F7') # Light Gray
COLOR_BORDER_LIGHT = colors.HexColor('#D1D1D6')
COLOR_WHITE = colors.white
COLOR_BLACK = colors.black

# Matplotlib colors (matching the palette)
MPL_COLOR_PRIMARY = '#007AFF'
MPL_COLOR_SUCCESS = '#34C759'
MPL_COLOR_WARNING = '#FF9500'
MPL_COLOR_ERROR = '#FF3B30'
MPL_COLOR_SECONDARY = '#5AC8FA' # Lighter blue for charts
MPL_COLOR_GRAY = '#8E8E93'
MPL_COLOR_TEXT_PRIMARY = '#1C1C1E' # Added for Matplotlib text

class ChecklistReportGenerator:
    """Generates a PDF report from checklist analysis results."""
    
    def __init__(self, json_file_path, output_pdf_path=None):
        """Initialize the report generator."""
        self.json_file_path = json_file_path
        
        if output_pdf_path is None:
            base_name = os.path.splitext(os.path.basename(json_file_path))[0]
            # Ensure output directory exists
            output_dir = "output"
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                except OSError as e:
                    print(f"Warning: Could not create output directory '{output_dir}'. Saving to current directory. Error: {e}")
                    output_dir = "."
            self.output_pdf_path = os.path.join(output_dir, f"{base_name}_report.pdf")
        else:
            self.output_pdf_path = output_pdf_path
            
        # Load data
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f: # Specify encoding
                self.results = json.load(f)
            if not isinstance(self.results, list): # Ensure results is a list
                 raise ValueError("JSON data is not a list of results.")
                
            if len(self.results) == 1 and isinstance(self.results[0], dict) and ('error' in self.results[0] or 'message' in self.results[0] or 'warning' in self.results[0]):
                self.error_message = self.results[0].get('error') or self.results[0].get('message') or self.results[0].get('warning')
                self.has_valid_data = False
            # Check if list contains only non-dict items or empty dicts
            elif not any(isinstance(item, dict) and item for item in self.results):
                self.error_message = "JSON file contains no valid check results."
                self.has_valid_data = False
                self.results = [] # Ensure results is empty list for consistency
            else:
                self.error_message = None
                self.has_valid_data = True
                # Filter out potential non-dict items just in case
                self.results = [r for r in self.results if isinstance(r, dict)] 
        except json.JSONDecodeError as e:
            self.results = []
            self.error_message = f"Error decoding JSON: {str(e)}"
            self.has_valid_data = False
        except Exception as e:
            self.results = []
            self.error_message = f"Error loading or parsing JSON data: {str(e)}"
            self.has_valid_data = False
            
        # --- Chart Styles (defined here for broader scope) ---
        self.chart_dpi = 200 # Increase DPI for clarity
        self.pie_chart_style = {'width': 3.3*inch, 'height': 2.5*inch}
        self.bar_chart_style = {'width': 3.3*inch, 'height': 2.5*inch}
        self.hbar_chart_style = {'width': 6.8*inch, 'height': 4.0*inch} # Branch chart needs width
        self.input_table_style = {'width': 6.6*inch}
        
        # --- Modernized Styles ---
        self.styles = getSampleStyleSheet()
        base_font = 'Helvetica'
        bold_font = 'Helvetica-Bold'
        italic_font = 'Helvetica-Oblique'

        self.title_style = ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['h1'],
            fontName=bold_font,
            fontSize=22,
            alignment=TA_CENTER,
            textColor=COLOR_TEXT_PRIMARY,
            spaceAfter=16 # Reduced space
        )
        self.subtitle_style = ParagraphStyle(
            name='ReportSubTitle',
            parent=self.styles['Normal'],
            fontName=base_font,
            fontSize=11,
            alignment=TA_CENTER,
            textColor=COLOR_TEXT_SECONDARY,
            spaceAfter=24
        )
        self.heading_style = ParagraphStyle(
            name='ReportHeading',
            parent=self.styles['h2'],
            fontName=bold_font,
            fontSize=14, # Slightly smaller heading
            textColor=COLOR_PRIMARY,
            spaceAfter=8,
            spaceBefore=18,
        )
        self.subheading_style = ParagraphStyle(
            name='ReportSubHeading',
            parent=self.styles['h3'],
            fontName=bold_font,
            fontSize=11, # Slightly smaller sub-heading
            textColor=COLOR_TEXT_PRIMARY,
            spaceAfter=6,
            spaceBefore=0 # Reduced space before box content
        )
        self.normal_style = ParagraphStyle(
            name='ReportNormal',
            parent=self.styles['Normal'],
            fontName=base_font,
            fontSize=9.5, # Slightly smaller base font
            textColor=COLOR_TEXT_PRIMARY,
            leading=13 # Adjusted Line spacing
        )
        self.normal_centered = ParagraphStyle(
            name='ReportNormalCentered', 
            parent=self.normal_style,
            alignment=TA_CENTER
        )
        self.small_gray_style = ParagraphStyle(
            name='SmallGray',
            parent=self.normal_style,
            fontSize=8.5, # Smaller gray text
            textColor=COLOR_TEXT_SECONDARY
        )
        self.label_style = ParagraphStyle(
            name='LabelStyle',
            parent=self.normal_style,
            fontName=bold_font,
            fontSize=8.5,
            textColor=COLOR_TEXT_SECONDARY
        )
        self.code_style = ParagraphStyle(
            name='CodeStyle',
            parent=self.normal_style,
            fontName='Courier',
            fontSize=9,
            textColor=COLOR_TEXT_PRIMARY,
            backColor=COLOR_BACKGROUND_LIGHT,
            borderPadding=5,
            # borderRadius=3 # Not supported directly
        )
        self.user_input_style = ParagraphStyle(
            name='UserInput',
            parent=self.normal_style,
            fontName=italic_font,
            fontSize=9, # Match normal text slightly better
            textColor=colors.darkblue,
            leftIndent=10, # Reduced indent
            borderLeftWidth=1.5,
            borderLeftColor=COLOR_BORDER_LIGHT,
            paddingLeft=6,
            spaceBefore=3,
            spaceAfter=3
        )
        
        # Extract phase info
        self.phase = "Analysis" # Default phase
        if self.has_valid_data and len(self.results) > 0:
            # Find first valid result with check_item and phase
            first_valid_result = next((r for r in self.results if isinstance(r, dict) and 'check_item' in r and 'phase' in r['check_item']), None)
            if first_valid_result:
                self.phase = first_valid_result['check_item']['phase']

    def _get_status_paragraph(self, is_met):
        if is_met:
            return Paragraph("✓ MET", ParagraphStyle('StatusMet', parent=self.normal_style, textColor=COLOR_SUCCESS, fontName='Helvetica-Bold', alignment=TA_CENTER))
        else:
            return Paragraph("✗ NOT MET", ParagraphStyle('StatusNotMet', parent=self.normal_style, textColor=COLOR_ERROR, fontName='Helvetica-Bold', alignment=TA_CENTER))

    def _get_reliability_paragraph(self, reliability):
        style = ParagraphStyle('ReliabilityScore', parent=self.normal_style, alignment=TA_CENTER)
        if reliability >= 90:
            style.textColor = COLOR_SUCCESS
        elif reliability >= 70:
            style.textColor = COLOR_PRIMARY
        elif reliability >= 50:
            style.textColor = COLOR_WARNING
        else:
            style.textColor = COLOR_ERROR
        return Paragraph(f"{reliability:.0f}%", style) # Use integer for cleaner look

    def _get_review_paragraph(self, needs_review):
        if needs_review:
            return Paragraph("YES", ParagraphStyle('ReviewYes', parent=self.normal_style, textColor=COLOR_WARNING, fontName='Helvetica-Bold', alignment=TA_CENTER))
        else:
            return Paragraph("NO", ParagraphStyle('ReviewNo', parent=self.normal_style, textColor=COLOR_TEXT_SECONDARY, alignment=TA_CENTER))
    
    def create_summary_table(self):
        """Create a minimalist summary table."""
        if not self.has_valid_data or not self.results:
            return None
            
        header = [
            Paragraph('ID', self.label_style),
            Paragraph('Check Name', self.label_style),
            Paragraph('Status', self.label_style),
            Paragraph('Reliability', self.label_style),
            Paragraph('Needs Review', self.label_style)
        ]
        data = [header]
        
        for result in self.results:
            if 'check_item' not in result:
                continue
                
            check_id = Paragraph(str(result['check_item']['id']), self.normal_style)
            check_name = Paragraph(result['check_item']['name'], self.normal_style)
            status = self._get_status_paragraph(result.get('is_met', False))
            reliability = self._get_reliability_paragraph(result.get('reliability', 0))
            needs_review = self._get_review_paragraph(result.get('needs_human_review', False))
            
            data.append([check_id, check_name, status, reliability, needs_review])
            
        if len(data) > 1:
            table = Table(data, colWidths=[0.4*inch, 3.6*inch, 0.8*inch, 0.9*inch, 0.9*inch]) # Adjusted widths
            
            style = TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LINEABOVE', (0, 0), (-1, 0), 1, COLOR_TEXT_PRIMARY), # Darker header top border
                ('LINEBELOW', (0, 0), (-1, 0), 0.5, COLOR_TEXT_PRIMARY), # Darker header bottom border
                ('LINEBELOW', (0, 1), (-1, -1), 0.25, COLOR_BORDER_LIGHT), # Lighter lines between rows
                ('ALIGN', (0, 0), (0, -1), 'CENTER'), # Center ID column
                ('ALIGN', (2, 0), (-1, -1), 'CENTER'), # Center Status, Reliability, Review
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6), # Padding header
                ('TOPPADDING', (0, 1), (-1, -1), 5), # Padding rows
                ('BOTTOMPADDING', (0, 1), (-1, -1), 5), # Padding rows
            ])
            table.setStyle(style)
            return table
        return None
        
    def create_charts(self):
        """Create modern charts for the report."""
        if not self.has_valid_data or not self.results:
            return []
            
        charts = []
        # Chart styles are now instance attributes (self.pie_chart_style, etc.)

        # --- 1. Status distribution --- 
        met_count = sum(1 for r in self.results if r.get('is_met', False))
        not_met_count = sum(1 for r in self.results if not r.get('is_met', False) and 'is_met' in r)
        if met_count > 0 or not_met_count > 0:
            fig, ax = plt.subplots(figsize=(4.5, 3)) # Compact size
            labels = [f'Met ({met_count})', f'Not Met ({not_met_count})']
            sizes = [met_count, not_met_count]
            colors = [MPL_COLOR_SUCCESS, MPL_COLOR_ERROR] 
            explode = (0.05, 0) # Slightly explode 'Met'
            
            # Remove autopct for cleaner look, legend provides counts
            wedges, texts = ax.pie(sizes, labels=None, colors=colors, 
                   startangle=90, pctdistance=0.85, explode=explode, wedgeprops={'width': 0.3})

            # Draw circle for donut chart effect - Handled by wedgeprops now
            ax.axis('equal')  
            plt.title('Status Distribution', fontsize=11, color=MPL_COLOR_TEXT_PRIMARY, pad=10)
            # Add legend below chart
            ax.legend(wedges, labels, title="Status", loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=8, ncol=2)

            plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout slightly for legend
            img_data = BytesIO()
            fig.savefig(img_data, format='png', bbox_inches='tight', dpi=self.chart_dpi) # Use instance DPI
            img_data.seek(0)
            charts.append(Image(img_data, **self.pie_chart_style))
            plt.close(fig)
            
        # --- 2. Reliability distribution --- 
        reliability_scores = [r.get('reliability', 0) for r in self.results if 'reliability' in r]
        if reliability_scores:
            fig, ax = plt.subplots(figsize=(4.8, 3.2)) # Slightly larger
            categories = ["< 50%", "50-69%", "70-89%", "90-100%"]
            counts = [
                sum(1 for s in reliability_scores if s < 50),
                sum(1 for s in reliability_scores if 50 <= s < 70),
                sum(1 for s in reliability_scores if 70 <= s < 90),
                sum(1 for s in reliability_scores if s >= 90)
            ]
            colors = [MPL_COLOR_ERROR, MPL_COLOR_WARNING, MPL_COLOR_SECONDARY, MPL_COLOR_SUCCESS]
            
            bars = ax.bar(categories, counts, color=colors)
            ax.set_ylabel('Number of Checks', fontsize=9, color=MPL_COLOR_GRAY)
            ax.set_title('Reliability Score Distribution', fontsize=11, color=MPL_COLOR_TEXT_PRIMARY, pad=10)
            ax.tick_params(axis='x', labelsize=8, colors=MPL_COLOR_GRAY)
            ax.tick_params(axis='y', labelsize=8, colors=MPL_COLOR_GRAY)
            ax.grid(axis='y', linestyle=':', alpha=0.6)
            ax.spines[['top', 'right']].set_visible(False) # Cleaner look

            # Add labels on top
            ax.bar_label(bars, padding=3, fontsize=8, color=MPL_COLOR_GRAY)
            ax.margins(y=0.1) # Add margin at the top

            plt.tight_layout()
            img_data = BytesIO()
            fig.savefig(img_data, format='png', bbox_inches='tight', dpi=self.chart_dpi)
            img_data.seek(0)
            charts.append(Image(img_data, **self.bar_chart_style))
            plt.close(fig)
            
        # --- 3. Branch distribution --- 
        branch_data = {}
        for r in self.results:
            if 'check_item' in r and 'branch_name' in r['check_item']:
                branch = r['check_item']['branch_name'] or 'Uncategorized'
                is_met = r.get('is_met', False)
                if branch not in branch_data: branch_data[branch] = {'met': 0, 'not_met': 0}
                branch_data[branch]['met' if is_met else 'not_met'] += 1
        
        if branch_data:
            # Sort branches by total count descending for better visualization
            sorted_branches = sorted(branch_data.items(), key=lambda item: sum(item[1].values()), reverse=True)
            branches = [item[0] for item in sorted_branches]
            met_counts = [item[1]['met'] for item in sorted_branches]
            not_met_counts = [item[1]['not_met'] for item in sorted_branches]
            
            # Increase base height and multiplier slightly for better spacing
            fig_height = max(3.2, 0.45 * len(branches)) # Dynamic height
            fig, ax = plt.subplots(figsize=(7.2, fig_height))
            
            y = range(len(branches))
            # Create stacked bars
            bars1 = ax.barh(y, met_counts, label='Met', color=MPL_COLOR_SUCCESS, height=0.6)
            bars2 = ax.barh(y, not_met_counts, left=met_counts, label='Not Met', color=MPL_COLOR_ERROR, height=0.6)
            
            ax.set_yticks(y)
            ax.set_yticklabels(branches, fontsize=8, color=MPL_COLOR_GRAY)
            ax.set_xlabel('Number of Checks', fontsize=9, color=MPL_COLOR_GRAY)
            ax.set_title('Status by Branch / Topic', fontsize=11, color=MPL_COLOR_TEXT_PRIMARY, pad=10)
            ax.tick_params(axis='x', labelsize=8, colors=MPL_COLOR_GRAY)
            ax.invert_yaxis() # Display top-to-bottom
            # Move legend outside plot area
            ax.legend(fontsize=8, frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
            ax.grid(axis='x', linestyle=':', alpha=0.6)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(axis='y', length=0) # Hide y-axis ticks

            # Add total count labels to the right
            for i, (met, not_met) in enumerate(zip(met_counts, not_met_counts)):
                total = met + not_met
                if total > 0:
                    # Adjust text position slightly based on total to avoid overlap
                    text_x_pos = total + (ax.get_xlim()[1] * 0.02) # Position relative to axis limit
                    ax.text(text_x_pos, i, str(total), va='center', ha='left', fontsize=8, color=MPL_COLOR_GRAY)

            # Adjust layout to make room for legend outside
            plt.tight_layout(rect=[0, 0, 0.9, 1], pad=0.5)
            img_data = BytesIO()
            fig.savefig(img_data, format='png', bbox_inches='tight', dpi=self.chart_dpi)
            img_data.seek(0)
            # Adjust image height based on number of branches
            img_height = max(2.8, 0.45 * len(branches) + 0.6)
            charts.append(Image(img_data, width=self.hbar_chart_style['width'], height=img_height*inch))
            plt.close(fig)
            
        # --- 4. Human Input Distribution --- 
        human_input_results = [r for r in self.results if r.get('user_input')]
        if human_input_results:
            # Combine Pie chart and Table logic
            total_checks = len(self.results)
            input_count = len(human_input_results)
            no_input_count = total_checks - input_count
            acceptance_inputs = sum(1 for r in human_input_results if r.get('user_input') in ["1", "skip", "continue", "accept"])
            info_provided = input_count - acceptance_inputs
            
            input_data_for_table = [
                [Paragraph('ID', self.label_style), 
                 Paragraph('Check Name', self.label_style), 
                 Paragraph('Input Type', self.label_style)]
            ]
            for r in human_input_results:
                input_type_text = "Accepted" if r.get('user_input') in ["1", "skip", "continue", "accept"] else "Provided Info"
                input_data_for_table.append([
                    Paragraph(str(r['check_item']['id']), self.small_gray_style),
                    Paragraph(r['check_item']['name'], self.small_gray_style),
                    Paragraph(input_type_text, self.small_gray_style)
                ])
            
            input_table = Table(input_data_for_table, 
                                colWidths=[0.4*inch, self.input_table_style['width'] - 0.4*inch - 1.3*inch, 1.3*inch])
            input_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LINEABOVE', (0, 0), (-1, 0), 1, COLOR_BORDER_LIGHT), 
                ('LINEBELOW', (0, 0), (-1, 0), 1, COLOR_BORDER_LIGHT), 
                ('LINEBELOW', (0, 1), (-1, -1), 0.25, COLOR_BORDER_LIGHT), # Light lines between rows
                ('BOTTOMPADDING', (0, 0), (-1, 0), 4), 
                ('TOPPADDING', (0, 1), (-1, -1), 3), 
                ('BOTTOMPADDING', (0, 1), (-1, -1), 3), 
                ('ALIGN', (0, 0), (0, -1), 'CENTER'), # Center ID
                ('ALIGN', (2, 0), (2, -1), 'LEFT'), # Align Input Type Left
            ]))
            
            # If there's space, maybe add a small pie chart next to the table
            # For now, just add the table if there was input
            if len(input_data_for_table) > 1:
                charts.append(Paragraph("Checks Receiving Human Input", self.subheading_style)) 
                charts.append(Spacer(1, 0.1*inch))
                charts.append(input_table)
            
        return charts
    
    def create_detailed_sections(self):
        """Create detailed sections for each check result within a box."""
        if not self.has_valid_data:
            return []
            
        elements = []
        # Style for the containing box table
        box_style = TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), COLOR_WHITE), # White background for box
            ('BOX', (0,0), (-1,-1), 0.5, COLOR_BORDER_LIGHT), # Light border
            #('ROUNDEDCORNERS', [6, 6, 6, 6]), # Not directly supported
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ])
        
        for result in self.results:
            if 'check_item' not in result:
                continue
                
            check_item = result['check_item']
            check_content = [] # Content for inside the box
            
            # --- Header Row within the box --- 
            status_p = self._get_status_paragraph(result.get('is_met', False))
            # Wrap header text
            header_p = Paragraph(f"<b>{check_item['id']}:</b> {check_item['name']}", self.subheading_style)
            
            header_table = Table([[header_p, status_p]], colWidths=['*', 1*inch]) # Give status fixed width
            header_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 0),
                ('RIGHTPADDING', (0,0), (-1,-1), 0),
                ('ALIGN', (1,0), (1,0), 'RIGHT'), # Align status right
                ]))
            check_content.append(header_table)
            check_content.append(Spacer(1, 0.08*inch))
            
            # --- Details Table (Label/Value pairs) --- 
            details_data = [
                [Paragraph("Description:", self.label_style), Paragraph(check_item['description'], self.normal_style)],
                [Paragraph("Branch:", self.label_style), Paragraph(f"{check_item.get('branch_name', 'N/A')} (ID: {check_item.get('branch_id', 'N/A')})", self.normal_style)],
                [Paragraph("Reliability:", self.label_style), self._get_reliability_paragraph(result.get('reliability', 0))],
                [Paragraph("Needs Review:", self.label_style), self._get_review_paragraph(result.get('needs_human_review', False))]
            ]
            details_table = Table(details_data, colWidths=[0.9*inch, '*']) # Fixed label width
            details_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 0),
                ('RIGHTPADDING', (0,0), (-1,-1), 0),
                ('TOPPADDING', (0,0), (-1,-1), 1),
                ('BOTTOMPADDING', (0,0), (-1,-1), 1),
                ('ALIGN', (1, 2), (1, 3), 'CENTER'), # Center reliability & review values
            ]))
            check_content.append(details_table)

            # --- Human Input (if available) ---
            user_input = result.get('user_input', None)
            if user_input:
                check_content.append(Spacer(1, 0.05*inch))
                if user_input in ["1", "skip", "continue", "accept"]:
                    input_label = Paragraph("<b>Human Input:</b> <i>User accepted assessment</i>", self.small_gray_style)
                    check_content.append(input_label)
                else:
                    input_label = Paragraph("<b>Human Input:</b>", self.label_style)
                    # Wrap user input paragraph
                    input_para = Paragraph(user_input, self.user_input_style)
                    check_content.append(input_label)
                    check_content.append(input_para)
            
            # --- Analysis Details ---
            analysis = result.get('analysis_details', None)
            if analysis:
                check_content.append(Spacer(1, 0.05*inch))
                check_content.append(Paragraph("Analysis Details:", self.label_style))
                check_content.append(Paragraph(analysis, self.normal_style))
            
            # --- Evidence Sources ---
            sources = result.get('sources', [])
            if sources:
                check_content.append(Spacer(1, 0.05*inch))
                check_content.append(Paragraph("Evidence Sources:", self.label_style))
                source_list = []
                for i, source in enumerate(sources, 1):
                    # Truncate long sources for display
                    display_source = (source[:180] + '...') if len(source) > 180 else source
                    # Add numbering
                    source_list.append(Paragraph(f"{i}. {display_source}", self.small_gray_style))
                check_content.extend(source_list)
            
            # --- Wrap content in the box Table ---
            # Use colWidths='*' to allow content to define width within the box
            box_table = Table([[check_content]], style=box_style, colWidths=['*'])
            elements.append(KeepTogether(box_table)) # Keep each box on one page if possible
            elements.append(Spacer(1, 0.15*inch)) # Reduced space between boxes
            
        return elements
    
    def generate_pdf(self):
        """Generate the PDF report."""
        # Use letter size for potentially wider content
        doc = SimpleDocTemplate(self.output_pdf_path, pagesize=letter, 
                                leftMargin=0.75*inch, rightMargin=0.75*inch, 
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        elements = []
        
        # --- Title Section --- 
        elements.append(Paragraph(f"Checklist Compliance Report", self.title_style))
        elements.append(Paragraph(f"Phase: {self.phase}", self.subtitle_style))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Generated: {timestamp}", self.small_gray_style))
        elements.append(Spacer(1, 0.3*inch))
        
        if not self.has_valid_data:
            elements.append(Paragraph("Error Report", self.heading_style))
            elements.append(Paragraph(f"Could not generate report from '{os.path.basename(self.json_file_path)}'.", self.normal_style))
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(f"<b>Error:</b> {self.error_message}", self.normal_style))
        else:
            # --- Executive Summary --- 
            elements.append(Paragraph("Executive Summary", self.heading_style))
            total_checks = len(self.results)
            met_count = sum(1 for r in self.results if r.get('is_met', False))
            not_met_count = total_checks - met_count
            meets_percentage = (met_count / total_checks * 100) if total_checks > 0 else 0
            review_count = sum(1 for r in self.results if r.get('needs_human_review', False))
            human_input_count = sum(1 for r in self.results if r.get('user_input'))
            acceptance_count = sum(1 for r in self.results if r.get('user_input') in ["1", "skip", "continue", "accept"])
            info_provided_count = human_input_count - acceptance_count
            
            # Improved summary paragraph
            summary_text = f"""
            This report summarizes the analysis of <b>{total_checks}</b> checklist items for the <b>{self.phase}</b> phase. 
            Overall, <b>{met_count} items ({meets_percentage:.1f}%)</b> were assessed as MET, while 
            <b>{not_met_count} items ({100-meets_percentage:.1f}%)</b> were assessed as NOT MET. 
            <b>{review_count} items</b> were flagged as requiring human review. 
            During the process, <b>{human_input_count} items</b> received human input, with 
            <b>{acceptance_count}</b> acceptances of the automated assessment and 
            <b>{info_provided_count}</b> instances where users provided additional details.
            """
            elements.append(Paragraph(summary_text, self.normal_style))
            elements.append(Spacer(1, 0.2*inch))
            
            # --- Summary Table --- 
            elements.append(Paragraph("Results Overview", self.heading_style))
            summary_table = self.create_summary_table()
            if summary_table:
                elements.append(summary_table)
            else:
                elements.append(Paragraph("No summary data available.", self.normal_style))
            elements.append(Spacer(1, 0.3*inch))
            
            # --- Charts and Visualizations --- 
            elements.append(Paragraph("Visualizations", self.heading_style))
            charts = self.create_charts()
            if charts:
                # Attempt side-by-side layout for pie/bar charts if both exist
                if len(charts) >= 2 and isinstance(charts[0], Image) and isinstance(charts[1], Image):
                     # Assuming first two are pie and reliability bar
                     # Increase spacing slightly and adjust colWidths to match new chart styles
                     spacer_width = 0.4*inch
                     chart_width = self.pie_chart_style['width'] # Use instance attribute
                     total_width = chart_width * 2 + spacer_width
                     chart_table = Table([[charts[0], Spacer(spacer_width, 0), charts[1]]], 
                                         colWidths=[chart_width, spacer_width, chart_width]) 
                     chart_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
                     elements.append(chart_table)
                     remaining_charts = charts[2:]
                else:
                     remaining_charts = charts

                # Add remaining charts sequentially
                for chart in remaining_charts:
                    elements.append(chart)
                    elements.append(Spacer(1, 0.2*inch))
            else:
                elements.append(Paragraph("No data available for charts.", self.normal_style))
            elements.append(Spacer(1, 0.3*inch))
                
            # --- Detailed Results --- 
            elements.append(Paragraph("Detailed Check Analysis", self.heading_style))
            detailed_sections = self.create_detailed_sections()
            if detailed_sections:
                elements.extend(detailed_sections)
            else:
                 elements.append(Paragraph("No detailed check results found.", self.normal_style))
            
        # --- Build PDF --- 
        try:
            doc.build(elements)
            print(f"PDF report generated: {self.output_pdf_path}")
            return self.output_pdf_path
        except Exception as e:
            print(f"Error building PDF: {e}")
            # Fallback: Create a simple text file with the error
            error_txt_path = self.output_pdf_path.replace(".pdf", "_error.txt") # Corrected assignment
            try:
                with open(error_txt_path, 'w', encoding='utf-8') as f: # Specify encoding
                    f.write(f"Failed to generate PDF report.\n")
                    f.write(f"JSON source: {self.json_file_path}\n")
                    f.write(f"Error: {e}\n")
                    if hasattr(self, 'results') and self.results: # Check if results exist
                        f.write("\n--- Results Data (first 5) ---\n")
                        try:
                            f.write(json.dumps(self.results[:5], indent=2))
                        except Exception as dump_e:
                            f.write(f"Could not dump results data: {dump_e}")
                    else:
                        f.write("\n--- No results data available ---\n")
                print(f"Error details saved to: {error_txt_path}")
            except Exception as write_e:
                 print(f"Could not write error file {error_txt_path}: {write_e}")
            return None

async def generate_pdf_report(results: list, output_pdf_path: str) -> str | None: # Added return type hint
    """Generate a PDF report from the analysis results.
    
    Args:
        results: List of analysis results
        output_pdf_path: Path where to save the PDF report

    Returns:
        The path to the generated PDF, or None if generation failed.
    """
    # Create a temporary JSON file with the results
    import tempfile
    import os
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_pdf_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Warning: Could not create output directory '{output_dir}'. Error: {e}")

    temp_json_path = ""
    pdf_path = None
    try:
        # Create a temporary file ensuring it's closed before ChecklistReportGenerator reads it
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_json:
            json.dump(results, temp_json, indent=2)
            temp_json_path = temp_json.name
        
        # Ensure the generator uses the specific output path provided
        generator = ChecklistReportGenerator(temp_json_path, output_pdf_path=output_pdf_path)
        pdf_path = generator.generate_pdf()

    except Exception as e:
        print(f"An error occurred during PDF generation process: {e}")
        # Optionally re-raise or handle more gracefully
    finally:
        # Clean up the temporary JSON file
        if temp_json_path and os.path.exists(temp_json_path):
            try:
                os.unlink(temp_json_path)
            except Exception as unlink_e:
                print(f"Error removing temporary file {temp_json_path}: {unlink_e}")
    return pdf_path # Return the path (or None if failed)

if __name__ == "__main__":
    # Get the latest results file in the current directory
    current_dir = '.'
    json_files = [f for f in os.listdir(current_dir) if f.startswith('analysis_results_') and f.endswith('.json')]
    if not json_files:
        print(f"No analysis results files (analysis_results_*.json) found in directory: {os.path.abspath(current_dir)}")
        exit(1)
        
    # Sort by modification time to get the most recent
    latest_file_path = max([os.path.join(current_dir, f) for f in json_files], key=os.path.getmtime)
    print(f"Generating report from file: {latest_file_path}")
    
    # Generate the report (output path will be determined by the generator)
    generator = ChecklistReportGenerator(latest_file_path)
    output_path = generator.generate_pdf()
    
    if output_path:
        print(f"Report generated successfully: {os.path.abspath(output_path)}") 
    else:
        print("Report generation failed.") 