import pandas as pd
import numpy as np
from collections import Counter
import re
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize
from pythainlp.corpus import thai_stopwords
from pythainlp.tag import pos_tag
from pythainlp.corpus import thai_words
import os
import json
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io

def setup_output_directories(base_dir):
    """Setup directory structure for analysis outputs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"dialect_analysis_{timestamp}")
    
    subdirs = {
        'plots': os.path.join(output_dir, 'plots'),
        'reports': os.path.join(output_dir, 'reports'),
        'stats': os.path.join(output_dir, 'stats')
    }
    
    for dir_path in subdirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return subdirs, output_dir

def save_plot(fig, filename, plots_dir):
    """Save plot to file"""
    filepath = os.path.join(plots_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Saved plot: {filepath}")
    return filepath

def save_json(data, filename, stats_dir):
    """Save data as JSON"""
    filepath = os.path.join(stats_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved stats: {filepath}")
    return filepath

def save_report(text, filename, reports_dir):
    """Save text report"""
    filepath = os.path.join(reports_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved report: {filepath}")
    return filepath

def analyze_dialect_distribution(df, plots_dir):
    """Analyze dialect distribution"""
    print("Analyzing dialect distribution...")
    dialect_dist = df['dialect'].value_counts()
    percentages = (dialect_dist / len(df) * 100).round(2)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='dialect')
    plt.title('การกระจายตัวของภาษาถิ่น')
    plt.xlabel('ภาษาถิ่น')
    plt.ylabel('จำนวนตัวอย่าง')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = save_plot(plt.gcf(), 'dialect_distribution.png', plots_dir)
    
    return {
        'distribution': {
            str(dialect): {
                'count': int(count),
                'percentage': f"{percentages[dialect]}%"
            } for dialect, count in dialect_dist.items()
        },
        'plot': plot_path
    }

def analyze_text_stats(df, plots_dir):
    """Analyze text statistics by dialect"""
    print("Analyzing text statistics...")
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].apply(lambda x: len(word_tokenize(x)))
    
    stats = {}
    for dialect in df['dialect'].unique():
        dialect_df = df[df['dialect'] == dialect]
        stats[dialect] = {
            'text_length': {
                'mean': f"{dialect_df['text_length'].mean():.1f}",
                'median': f"{dialect_df['text_length'].median():.1f}",
                'min': f"{dialect_df['text_length'].min():.1f}",
                'max': f"{dialect_df['text_length'].max():.1f}"
            },
            'word_count': {
                'mean': f"{dialect_df['word_count'].mean():.1f}",
                'median': f"{dialect_df['word_count'].median():.1f}",
                'min': f"{dialect_df['word_count'].min():.1f}",
                'max': f"{dialect_df['word_count'].max():.1f}"
            }
        }
    
    # Create boxplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    sns.boxplot(data=df, x='dialect', y='text_length', ax=ax1)
    ax1.set_title('การกระจายความยาวข้อความตามภาษาถิ่น')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    sns.boxplot(data=df, x='dialect', y='word_count', ax=ax2)
    ax2.set_title('การกระจายจำนวนคำตามภาษาถิ่น')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plot_path = save_plot(fig, 'text_statistics.png', plots_dir)
    
    return stats, plot_path

def analyze_common_words(df, plots_dir):
    """Analyze common words by dialect"""
    print("Analyzing common words...")
    stopwords = thai_stopwords()
    common_words = {}
    
    for dialect in df['dialect'].unique():
        dialect_texts = df[df['dialect'] == dialect]['text']
        words = []
        for text in dialect_texts:
            tokens = word_tokenize(text)
            words.extend([w for w in tokens if w not in stopwords and len(w) > 1])
        
        word_freq = Counter(words).most_common(20)
        common_words[dialect] = {word: count for word, count in word_freq}
        
        # Create word frequency plot for each dialect
        plt.figure(figsize=(12, 6))
        words, counts = zip(*word_freq)
        plt.bar(words, counts)
        plt.title(f'คำที่พบบ่อยในภาษา{dialect}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = save_plot(plt.gcf(), f'common_words_{dialect}.png', plots_dir)
        common_words[dialect]['plot'] = plot_path
    
    return common_words

def generate_pdf_report(results, output_path):
    """Generate comprehensive PDF report"""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Add title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("รายงานการวิเคราะห์คุณภาพข้อมูลภาษาถิ่น", title_style))
    story.append(Spacer(1, 20))
    
    # Add basic info
    basic_info = [
        ["ไฟล์ข้อมูล:", results['dataset_info']['filename']],
        ["จำนวนตัวอย่าง:", str(results['dataset_info']['total_samples'])],
        ["วันที่วิเคราะห์:", results['dataset_info']['timestamp']]
    ]
    
    tbl = Table(basic_info, colWidths=[120, 300])
    tbl.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12)
    ]))
    story.append(tbl)
    story.append(Spacer(1, 20))
    
    # Add analysis sections
    for analysis_name, analysis_data in results['analyses'].items():
        # Add section header
        story.append(Paragraph(analysis_name.replace('_', ' ').title(), styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Add plots
        if 'plot' in analysis_data:
            img = Image(analysis_data['plot'])
            img.drawHeight = 300
            img.drawWidth = 450
            story.append(img)
            story.append(Spacer(1, 12))
        elif 'plots' in analysis_data:
            for plot_path in analysis_data['plots'].values():
                img = Image(plot_path)
                img.drawHeight = 300
                img.drawWidth = 450
                story.append(img)
                story.append(Spacer(1, 12))
        
        # Add statistics
        if isinstance(analysis_data, dict):
            stats_data = []
            for key, value in analysis_data.items():
                if key not in ['plot', 'plots']:
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if not isinstance(subvalue, dict):
                                stats_data.append([f"{key} - {subkey}:", str(subvalue)])
                    else:
                        stats_data.append([key + ":", str(value)])
            
            if stats_data:
                tbl = Table(stats_data, colWidths=[200, 220])
                tbl.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 12)
                ]))
                story.append(tbl)
                story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    print(f"PDF report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Check dialect dataset quality')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                      help='Base directory for analysis outputs')
    
    try:
        args = parser.parse_args()
        
        print(f"\nAnalyzing dialect dataset: {args.input_file}")
        print("=" * 50)
        
        # Setup output directories
        print(f"\nSetting up output directories in {args.output_dir}...")
        dirs, output_dir = setup_output_directories(args.output_dir)
        
        # Load dataset
        print(f"\nLoading dataset...")
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found: {args.input_file}")
            return
        
        df = pd.read_csv(args.input_file)
        print(f"Successfully loaded {len(df)} samples")
        
        # Run analyses
        results = {
            'dataset_info': {
                'filename': args.input_file,
                'total_samples': len(df),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'analyses': {}
        }
        
        # Dialect distribution
        dist_results = analyze_dialect_distribution(df, dirs['plots'])
        results['analyses']['dialect_distribution'] = dist_results
        
        # Text statistics by dialect
        stats_results, stats_plot = analyze_text_stats(df, dirs['plots'])
        results['analyses']['text_statistics'] = {
            'stats': stats_results,
            'plot': stats_plot
        }
        
        # Common words by dialect
        common_words = analyze_common_words(df, dirs['plots'])
        results['analyses']['common_words'] = common_words
        
        # Save final results
        print("\nSaving analysis results...")
        save_json(results, 'analysis_results.json', dirs['stats'])
        
        # Generate PDF report
        print("\nGenerating PDF report...")
        pdf_path = os.path.join(output_dir, 'dialect_quality_report.pdf')
        generate_pdf_report(results, pdf_path)
        
        print("\nAnalysis complete!")
        print("=" * 50)
        print(f"Results saved in: {output_dir}")
        print(f"- Full PDF Report: {pdf_path}")
        print(f"- Individual files:")
        print(f"  - Plots: {dirs['plots']}")
        print(f"  - Report: {dirs['reports']}")
        print(f"  - Statistics: {dirs['stats']}")
        
    except Exception as e:
        print(f"\nError during analysis:")
        import traceback
        print(traceback.format_exc())
        return

if __name__ == "__main__":
    main()
