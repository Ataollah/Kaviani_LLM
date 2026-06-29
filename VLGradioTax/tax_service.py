from database import find_citizen_by_national_code, get_citizen_exemptions, save_tax_record, get_tax_history
from datetime import datetime

TAX_BRACKETS = [
    (500000000, 0.0),
    (1000000000, 0.10),
    (1500000000, 0.15),
    (2000000000, 0.20),
    (float('inf'), 0.25),
]

MINIMUM_WAGE_EXEMPTION = 500000000

def calculate_base_tax(income, tax_rate):
    taxable_income = max(0, income - MINIMUM_WAGE_EXEMPTION)
    
    if taxable_income == 0:
        return 0
    
    tax = 0
    previous_bracket = 0
    
    for bracket_limit, bracket_rate in TAX_BRACKETS:
        if taxable_income <= previous_bracket:
            break
        
        taxable_in_bracket = min(taxable_income, bracket_limit) - previous_bracket
        if taxable_in_bracket > 0:
            tax += taxable_in_bracket * bracket_rate
        
        previous_bracket = bracket_limit
    
    job_tax = taxable_income * tax_rate
    
    final_tax = (tax + job_tax) / 2
    
    return final_tax

def apply_exemptions(base_tax, exemptions):
    total_exemption_percentage = sum(e['percentage'] for e in exemptions)
    total_exemption_percentage = min(total_exemption_percentage, 100)
    
    final_tax = base_tax * (1 - total_exemption_percentage / 100)
    
    return max(0, final_tax)

def calculate_tax(citizen_data):
    income = citizen_data.get('annual_income', 0)
    tax_rate = citizen_data.get('tax_rate', 0.15)
    citizen_id = citizen_data.get('id')
    
    base_tax = calculate_base_tax(income, tax_rate)
    
    exemptions = get_citizen_exemptions(citizen_id) if citizen_id else []
    
    final_tax = apply_exemptions(base_tax, exemptions)
    
    current_year = datetime.now().year
    if citizen_id and final_tax > 0:
        save_tax_record(citizen_id, current_year, income, final_tax)
    
    return {
        'income': income,
        'base_tax': base_tax,
        'exemptions': exemptions,
        'total_exemption_percentage': sum(e['percentage'] for e in exemptions),
        'final_tax': final_tax,
        'year': current_year
    }

def format_currency(amount):
    return "{:,.0f}".format(amount)

def generate_tax_report(citizen_data, tax_result):
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    گزارش محاسبه مالیات                        ║
╠══════════════════════════════════════════════════════════════╣
║  اطلاعات شخصی                                                 ║
╠══════════════════════════════════════════════════════════════╣
║  نام و نام خانوادگی: {citizen_data['first_name']} {citizen_data['last_name']}
║  کد ملی: {citizen_data['national_code']}
║  نام پدر: {citizen_data.get('father_name', '-')}
║  محل سکونت: {citizen_data.get('city_name', '-')} - {citizen_data.get('province_name', '-')}
║  شغل: {citizen_data.get('job_name', '-')}
╠══════════════════════════════════════════════════════════════╣
║  اطلاعات مالی                                                 ║
╠══════════════════════════════════════════════════════════════╣
║  درآمد سالانه: {format_currency(tax_result['income'])} ریال
║  مالیات پایه: {format_currency(tax_result['base_tax'])} ریال
"""
    
    if tax_result['exemptions']:
        report += "╠══════════════════════════════════════════════════════════════╣\n"
        report += "║  معافیت‌ها                                                    ║\n"
        for exemption in tax_result['exemptions']:
            report += f"║  - {exemption['name']}: {exemption['percentage']}%\n"
        report += f"║  مجموع معافیت: {tax_result['total_exemption_percentage']}%\n"
    
    report += f"""╠══════════════════════════════════════════════════════════════╣
║  مالیات نهایی قابل پرداخت                                     ║
╠══════════════════════════════════════════════════════════════╣
║  مبلغ: {format_currency(tax_result['final_tax'])} ریال
║  سال مالی: {tax_result['year']}
╚══════════════════════════════════════════════════════════════╝
"""
    
    return report