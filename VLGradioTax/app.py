import gradio as gr
from graph import run_tax_calculation
from database import find_citizen_by_national_code, get_tax_history
from tax_service import format_currency
from ocr_service import get_model_choices, get_model_labels, AVAILABLE_MODELS

def process_tax_request(image):
    
    result = run_tax_calculation(image)
    return result

def search_citizen(national_code):
    if not national_code or len(national_code) != 10:
        return "❌ لطفاً کد ملی ۱۰ رقمی وارد کنید."
    
    citizen = find_citizen_by_national_code(national_code)
    
    if not citizen:
        return "❌ کاربری با این کد ملی یافت نشد."
    
    info = f"""
╔══════════════════════════════════════════════════════════════╗
║                    اطلاعات کاربر                              ║
╠══════════════════════════════════════════════════════════════╣
║  نام: {citizen['first_name']} {citizen['last_name']}
║  کد ملی: {citizen['national_code']}
║  نام پدر: {citizen.get('father_name', '-')}
║  تاریخ تولد: {citizen.get('birth_date', '-')}
║  شهر: {citizen.get('city_name', '-')}
║  استان: {citizen.get('province_name', '-')}
║  شغل: {citizen.get('job_name', '-')}
║  نرخ مالیات شغلی: {citizen.get('tax_rate', 0) * 100}%
║  درآمد سالانه: {format_currency(citizen.get('annual_income', 0))} ریال
║  وضعیت: {'فعال ✅' if citizen.get('is_active') else 'غیرفعال ❌'}
╚══════════════════════════════════════════════════════════════╝
"""
    
    history = get_tax_history(citizen['id'])
    if history:
        info += "\n\n📋 سوابق مالیاتی:\n"
        info += "─" * 50 + "\n"
        for record in history:
            status = "✅ پرداخت شده" if record['is_paid'] else "⏳ در انتظار پرداخت"
            info += f"سال {record['year']}: مالیات {format_currency(record['tax_amount'])} ریال - {status}\n"
    
    return info

def create_demo_interface():
    model_choices = [(label, model_id) for model_id, label in AVAILABLE_MODELS]
    
    with gr.Blocks(
        title="سیستم محاسبه مالیات",
        theme=gr.themes.Soft(),
        css="""
        .rtl {direction: rtl; text-align: right;}
        .output-text {font-family: 'Courier New', monospace; white-space: pre-wrap;}
        .model-dropdown {min-width: 300px;}
        """
    ) as demo:
        
        gr.Markdown("""
        # 🧮 سیستم محاسبه مالیات
        ### به سیستم هوشمند محاسبه مالیات خوش آمدید
        
        این سیستم با استفاده از هوش مصنوعی (OpenRouter)، اطلاعات کارت ملی شما را استخراج کرده و مالیات شما را محاسبه می‌کند.
        """, elem_classes="rtl")
        
        with gr.Tabs():
            with gr.TabItem("📷 محاسبه مالیات"):
                with gr.Row():
                    with gr.Column(scale=1):
                        
                        image_input = gr.Image(
                            label="تصویر کارت ملی",
                            type="filepath",
                            elem_classes="rtl"
                        )
                        
                        calculate_btn = gr.Button(
                            "🔄 محاسبه مالیات",
                            variant="primary"
                        )
                    
                    with gr.Column(scale=2):
                        output = gr.Textbox(
                            label="نتیجه",
                            lines=25,
                            elem_classes=["rtl", "output-text"]
                        )
                
                calculate_btn.click(
                    fn=process_tax_request,
                    inputs=[image_input],
                    outputs=output
                )
            
            with gr.TabItem("🔍 جستجوی کاربر"):
                with gr.Row():
                    with gr.Column(scale=1):
                        search_input = gr.Textbox(
                            label="کد ملی",
                            placeholder="کد ملی ۱۰ رقمی را وارد کنید",
                            max_lines=1,
                            elem_classes="rtl"
                        )
                        
                        search_btn = gr.Button(
                            "🔍 جستجو",
                            variant="primary"
                        )
                        
                        gr.Markdown("""
                        ### کدهای ملی نمونه برای تست:
                        - `0012345678` - علی محمدی (کارمند دولتی)
                        - `0023456789` - مریم احمدی (پزشک)
                        - `0034567890` - محمد رضایی (مهندس)
                        - `0045678901` - زهرا حسینی (بازنشسته)
                        - `0056789012` - امیر کریمی (آزاد)
                        """, elem_classes="rtl")
                    
                    with gr.Column(scale=2):
                        search_output = gr.Textbox(
                            label="نتیجه جستجو",
                            lines=20,
                            elem_classes=["rtl", "output-text"]
                        )
                
                search_btn.click(
                    fn=search_citizen,
                    inputs=search_input,
                    outputs=search_output
                )
                
                search_input.submit(
                    fn=search_citizen,
                    inputs=search_input,
                    outputs=search_output
                )
            
            with gr.TabItem("📖 راهنما"):
                gr.Markdown("""
                ## راهنمای استفاده از سیستم
                
                ### 🔑 دریافت کلید API از OpenRouter
                1. به سایت [OpenRouter.ai](https://openrouter.ai) بروید
                2. یک حساب کاربری بسازید
                3. از بخش API Keys یک کلید جدید ایجاد کنید
                4. کلید را در فیلد مربوطه وارد کنید
                
                ### 📷 محاسبه مالیات با کارت ملی
                1. کلید API خود از OpenRouter را وارد کنید
                2. مدل هوش مصنوعی مورد نظر را انتخاب کنید
                3. تصویر کارت ملی خود را آپلود کنید
                4. دکمه "محاسبه مالیات" را بزنید
                5. سیستم به صورت خودکار کد ملی را استخراج و مالیات را محاسبه می‌کند
                
                ### 🤖 مدل‌های پشتیبانی شده
                - **GPT-4o**: بهترین کیفیت برای تشخیص تصویر
                - **GPT-4o Mini**: سریع‌تر و ارزان‌تر
                - **Claude 3.5 Sonnet**: کیفیت بالا از Anthropic
                - **Gemini Pro Vision**: مدل گوگل
                - **Llama 3.2 Vision**: مدل متن‌باز Meta
                
                ### 🔍 جستجوی کاربر
                - می‌توانید با وارد کردن کد ملی، اطلاعات کاربر را مشاهده کنید
                - سوابق مالیاتی کاربر نیز نمایش داده می‌شود
                
                ### 📊 نحوه محاسبه مالیات
                - معافیت پایه: ۵۰۰ میلیون ریال
                - نرخ مالیات بر اساس شغل و پلکان درآمدی محاسبه می‌شود
                - معافیت‌های ویژه (معلولیت، ایثارگری و ...) اعمال می‌شود
                
                ### ⚠️ نکات مهم
                - تصویر کارت ملی باید واضح و خوانا باشد
                - کد ملی باید ۱۰ رقم باشد
                - برای استفاده از قابلیت OCR، کلید API معتبر نیاز است
                - هزینه استفاده از API بر اساس مدل انتخابی متفاوت است
                """, elem_classes="rtl")
            
            with gr.TabItem("ℹ️ درباره"):
                gr.Markdown("""
                ## درباره سیستم
                
                این سیستم با استفاده از تکنولوژی‌های زیر ساخته شده است:
                
                ### 🛠️ تکنولوژی‌ها
                - **LangChain**: فریم‌ورک برای کار با مدل‌های زبانی
                - **LangGraph**: برای مدیریت گراف پردازش
                - **OpenRouter**: دسترسی به مدل‌های مختلف AI
                - **Gradio**: رابط کاربری وب
                - **SQLite**: پایگاه داده
                
                ### 📋 ساختار پایگاه داده
                - **provinces**: استان‌ها
                - **cities**: شهرها
                - **job_categories**: دسته‌بندی مشاغل
                - **citizens**: شهروندان
                - **tax_records**: سوابق مالیاتی
                - **tax_exemptions**: معافیت‌های مالیاتی
                - **citizen_exemptions**: معافیت‌های شهروندان
                
                ### 🔄 گردش کار
                ```
                شروع → دریافت تصویر → استخراج کد ملی → جستجو در پایگاه داده → محاسبه مالیات → نمایش نتیجه
                ```
                """, elem_classes="rtl")
        
        gr.Markdown("""
        ---
        🔗 [OpenRouter](https://openrouter.ai) | 📚 [LangChain](https://langchain.com) | 🎨 [Gradio](https://gradio.app)
        """, elem_classes="rtl")
    
    return demo

if __name__ == "__main__":
    demo = create_demo_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=1689,
        share=False
    )