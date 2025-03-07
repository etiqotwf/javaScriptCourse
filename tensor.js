// استيراد مكتبة TensorFlow.js لاستخدامها في بناء النماذج والتعامل مع البيانات
import * as tf from '@tensorflow/tfjs';

// استيراد وحدة 'fs/promises' من Node.js للتعامل مع نظام الملفات بطريقة غير متزامنة
import fs from 'fs/promises';

// استيراد وحدة 'readline' من Node.js لإنشاء واجهة لقراءة الإدخال من المستخدم عبر سطر الأوامر
import readline from 'readline';

// استيراد createCanvas من مكتبة 'canvas' لإنشاء ورسم الرسوم البيانية بدون الحاجة إلى مستعرض
import { createCanvas } from 'canvas';

// استيراد مكتبة Chart.js لرسم الرسوم البيانية بسهولة
import Chart from 'chart.js/auto';

// استيراد الدالة 'exec' من وحدة 'child_process' لتشغيل أوامر النظام في بيئة Node.js
import { exec } from 'child_process';

// استيراد مكتبة 'xlsx' للتعامل مع ملفات Excel بصيغة XLSX (قراءة وكتابة البيانات)
import * as XLSX from 'xlsx';

// تحديد مسار حفظ نموذج الذكاء الاصطناعي في ملف JSON
const MODEL_PATH = './storage/model.json';

// تحديد مسار حفظ الرسم البياني المُنشأ
const CHART_PATH = './price_chart.png';

// تحديد مسار ملف Excel الذي يحتوي على العمليات أو البيانات المخزنة
const EXCEL_PATH = './operations.xlsx';

// تحديد سعر الصرف ليتم استخدامه في العمليات الحسابية داخل التطبيق
const EXCHANGE_RATE = 52;

/**
 * دالة لتطبيع البيانات، بحيث يتم تحويل القيم إلى نطاق بين 0 و 1 
 * مما يساعد في تحسين أداء النموذج عند التدريب
 * @param {Array} data - المصفوفة التي تحتوي على البيانات الأصلية
 * @param {number} min - الحد الأدنى للقيم في البيانات
 * @param {number} max - الحد الأقصى للقيم في البيانات
 * @returns {Array} - مصفوفة تحتوي على القيم بعد التطبيع
 */
function normalizeData(data, min, max) {
    return data.map(value => (value - min) / (max - min));
}

/**
 * دالة لإزالة التطبيع من البيانات المُعالجة سابقًا، حيث يتم إرجاع القيم إلى نطاقها الأصلي
 * بعد أن كانت بين 0 و 1
 * @param {Array} data - المصفوفة التي تحتوي على القيم المُطبّعة
 * @param {number} min - الحد الأدنى للقيم الأصلية
 * @param {number} max - الحد الأقصى للقيم الأصلية
 * @returns {Array} - مصفوفة تحتوي على القيم بعد إلغاء التطبيع
 */
function denormalizeData(data, min, max) {
    return data.map(value => value * (max - min) + min);
}

/**
 * دالة لحفظ النموذج المدرب داخل ملف JSON حتى يمكن إعادة استخدامه لاحقًا
 * @param {tf.LayersModel} model - النموذج الذي سيتم حفظه
 */
async function saveModel(model) {
    await model.save(tf.io.withSaveHandler(async (artifacts) => {
        // تحويل كائن النموذج إلى JSON ثم حفظه في ملف
        await fs.writeFile(MODEL_PATH, JSON.stringify(artifacts));
        console.log("💾 Model saved successfully!"); // طباعة رسالة نجاح عند الحفظ
    }));
}

/**
 * دالة لتحميل النموذج من ملف JSON إذا كان متوفرًا، وإذا لم يكن موجودًا، يتم إنشاء نموذج جديد
 * @returns {tf.LayersModel | null} - إما أن يتم إرجاع النموذج المحمّل أو null في حال عدم وجود نموذج محفوظ
 */
async function loadModel() {
    try {
        // التأكد مما إذا كان ملف النموذج موجودًا قبل تحميله
        await fs.access(MODEL_PATH);
        console.log("📂 Loading existing model...");

        // قراءة البيانات من ملف النموذج وتحويلها إلى كائن JSON
        const rawData = await fs.readFile(MODEL_PATH, 'utf-8');
        const modelArtifacts = JSON.parse(rawData);

        // تحميل النموذج إلى TensorFlow.js من البيانات المخزنة في الذاكرة
        return await tf.loadLayersModel(tf.io.fromMemory(modelArtifacts));
    } catch {
        console.log("⚠️ No saved model found, creating a new one...");
        return null; // إرجاع null في حال عدم العثور على نموذج محفوظ
    }
}


/**
 * دالة لإنشاء نموذج الشبكات العصبية الاصطناعية باستخدام TensorFlow.js
 * يقوم هذا النموذج بتوقع الأسعار بناءً على 4 مدخلات
 */
function createModel() {
    const model = tf.sequential(); // إنشاء نموذج تسلسلي (Sequential)

    // إضافة الطبقة الأولى، وهي طبقة كثيفة (Dense) تحتوي على 10 خلايا عصبية
    model.add(tf.layers.dense({ 
        units: 10,               // عدد الخلايا العصبية في الطبقة
        activation: 'relu',      // استخدام دالة التنشيط ReLU لتحسين أداء النموذج
        inputShape: [4]          // تحديد عدد المدخلات (4 ميزات لكل بيانات)
    }));

    // إضافة الطبقة الأخيرة، وهي طبقة خرج تحتوي على خلية عصبية واحدة
    model.add(tf.layers.dense({ 
        units: 1                 // طبقة تحتوي على خلية واحدة لتوليد التوقع
    }));

    // تجميع (Compile) النموذج مع تحديد:
    // - المُحسّن 'adam' لتحسين أداء النموذج أثناء التدريب
    // - دالة الخطأ 'meanSquaredError' لحساب الخطأ بين القيم الفعلية والتوقعات
    model.compile({ 
        optimizer: 'adam', 
        loss: 'meanSquaredError' 
    });

    console.log("🏗️ New model created!"); // طباعة رسالة تفيد بإنشاء النموذج بنجاح
    return model; // إرجاع النموذج لاستخدامه في التدريب
}

/**
 * دالة لتدريب النموذج العصبي باستخدام بيانات الإدخال والمخرجات
 * @param {tf.LayersModel} model - النموذج الذي سيتم تدريبه
 * @returns {Object} - القيم الدنيا والعليا للمدخلات والمخرجات لاستخدامها في إزالة التطبيع لاحقًا
 */
async function trainModel(model) {
    console.log("🏋️ Training the model..."); // طباعة رسالة تفيد ببدء عملية التدريب

    // بيانات الإدخال (المدخلات) التي سيتم تدريب النموذج عليها
    const rawXs = [
        [120, 3, 2, 15],  // مثال لبيانات: مساحة العقار، عدد الغرف، عدد الحمامات، عمر العقار
        [200, 4, 3, 5],
        [150, 3, 2, 10],
        [180, 4, 3, 8]
    ];

    // القيم المقابلة لبيانات الإدخال (المخرجات)، وهي الأسعار الفعلية للعقارات
    const rawYs = [[500000], [800000], [600000], [750000]];

    // استخراج القيم الدنيا والعليا من بيانات الإدخال والمخرجات لتطبيع البيانات
    const minInput = Math.min(...rawXs.flat());  // أصغر قيمة بين جميع المدخلات
    const maxInput = Math.max(...rawXs.flat());  // أكبر قيمة بين جميع المدخلات
    const minOutput = Math.min(...rawYs.flat()); // أصغر قيمة بين جميع المخرجات
    const maxOutput = Math.max(...rawYs.flat()); // أكبر قيمة بين جميع المخرجات

    // تحويل بيانات الإدخال إلى شكل ملائم للتدريب بعد تطبيعها إلى نطاق (0-1)
    const xs = tf.tensor2d(normalizeData(rawXs.flat(), minInput, maxInput), [4, 4]);

    // تحويل بيانات المخرجات إلى شكل مناسب للتدريب بعد التطبيع
    const ys = tf.tensor2d(normalizeData(rawYs.flat(), minOutput, maxOutput), [4, 1]);

    // تدريب النموذج باستخدام البيانات المتاحة لعدد معين من الدورات (epochs)
    await model.fit(xs, ys, { epochs: 200 });

    console.log("✅ Training completed!"); // طباعة رسالة تفيد بانتهاء عملية التدريب

    // حفظ النموذج بعد التدريب ليتم استخدامه لاحقًا دون الحاجة لإعادة التدريب
    await saveModel(model);

    // إرجاع القيم الدنيا والعليا لاستخدامها لاحقًا في إزالة التطبيع عند التوقع
    return { minInput, maxInput, minOutput, maxOutput };
}

/**
 * دالة لطلب المدخلات من المستخدم عبر سطر الأوامر
 * يقوم المستخدم بإدخال معلومات عن العقار مثل المساحة وعدد الغرف والحمامات والعمر
 * @returns {Promise<number[]>} - مصفوفة تحتوي على المدخلات بعد تحويلها إلى أرقام
 */
async function askUserForInputs() {
    // إنشاء واجهة إدخال للإدخال والإخراج عبر سطر الأوامر
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

    // دالة مساعدة لطرح الأسئلة على المستخدم وانتظار إدخال الإجابة
    function askQuestion(question) {
        return new Promise(resolve => rl.question(question, resolve));
    }

    // طلب إدخال البيانات من المستخدم
    const area = await askQuestion("📏 Enter house area (square meters): ");  // المساحة بالمتر المربع
    const rooms = await askQuestion("🛏️ Enter number of rooms: ");           // عدد الغرف
    const bathrooms = await askQuestion("🚿 Enter number of bathrooms: ");    // عدد الحمامات
    const age = await askQuestion("📅 Enter house age (years): ");            // عمر العقار بالسنوات

    // إغلاق واجهة الإدخال بعد الانتهاء من جمع البيانات
    rl.close();

    // تحويل البيانات المدخلة إلى أرقام وإرجاعها كمصفوفة
    return [parseFloat(area), parseInt(rooms), parseInt(bathrooms), parseInt(age)];
}

/**
 * دالة لإنشاء رسم بياني يعرض بيانات العقار مع السعر المتوقع
 * @param {number[]} userInputs - المدخلات التي أدخلها المستخدم (المساحة، الغرف، الحمامات، العمر)
 * @param {number} priceUSD - السعر المتوقع بالدولار الأمريكي
 * @param {number} priceEGP - السعر المتوقع بالجنيه المصري
 */
async function generateChart(userInputs, priceUSD, priceEGP) {
    // تحديد أبعاد الرسم البياني
    const width = 600, height = 400;

    // إنشاء كائن الرسم (Canvas)
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext('2d');

    // إنشاء الرسم البياني باستخدام مكتبة Chart.js
    new Chart(ctx, {
        type: 'bar', // نوع الرسم البياني: أعمدة
        data: {
            labels: ['📏 Area', '🛏️ Rooms', '🚿 Bathrooms', '📅 Age'], // أسماء المحاور
            datasets: [{
                label: 'House Parameters', // اسم البيانات
                data: userInputs, // القيم المدخلة من قبل المستخدم
                backgroundColor: ['#36A2EB', '#FFCE56', '#4CAF50', '#FF6384'] // ألوان الأعمدة
            }]
        },
        options: {
            responsive: false, // تعطيل استجابة الحجم التلقائي لضمان بقاء الرسم البياني بالحجم المطلوب
            plugins: {
                title: {
                    display: true,
                    text: `Estimated House Price: $${priceUSD.toFixed(2)} (~ EGP ${priceEGP.toFixed(2)})` // عنوان الرسم البياني
                },
                legend: { display: false }, // إخفاء مفتاح البيانات
                tooltip: { enabled: true }  // تفعيل تلميحات البيانات عند تمرير الفأرة
            }
        }
    });

    // تحويل الرسم البياني إلى صورة بصيغة PNG
    const buffer = canvas.toBuffer('image/png');

    // حفظ الصورة في ملف محلي
    await fs.writeFile(CHART_PATH, buffer);
    console.log(`📊 Chart saved as ${CHART_PATH}`); // طباعة رسالة تفيد بحفظ الصورة

    // فتح الصورة تلقائيًا بعد حفظها
    exec(`start ${CHART_PATH}`, (err) => {
        if (err) console.error("⚠️ Failed to open the chart:", err);
    });
}

async function logToExcel(userInputs, priceUSD, priceEGP) {
    try {
        let workbook;
        try {
            // 📂 محاولة قراءة ملف Excel إذا كان موجودًا
            const fileBuffer = await fs.readFile(EXCEL_PATH);
            workbook = XLSX.read(fileBuffer, { type: "buffer" });
        } catch {
            // 📄 إذا لم يكن الملف موجودًا، نقوم بإنشاء ملف جديد وإضافة رأس الجدول
            workbook = XLSX.utils.book_new();
            const sheet = XLSX.utils.aoa_to_sheet([
                ["📅 Date & Time", "📏 Area", "🛏️ Rooms", "🚿 Bathrooms", "📅 Age", "💲 Price (USD)", "💰 Price (EGP)"]
            ]);
            XLSX.utils.book_append_sheet(workbook, sheet, "Operations");
        }

        // 📜 الحصول على الورقة التي تحتوي على البيانات
        const sheet = workbook.Sheets["Operations"];
        
        // 📝 إعداد صف البيانات الجديد الذي سيتم إضافته إلى Excel
        const newRow = [
            new Date().toLocaleString(), // 🕒 حفظ التاريخ والوقت الحالي
            ...userInputs, // 📏 القيم التي أدخلها المستخدم: المساحة، عدد الغرف، عدد الحمامات، وعمر العقار
            priceUSD.toFixed(2), // 💲 السعر بالدولار
            priceEGP.toFixed(2)   // 💰 السعر بالجنيه المصري
        ];

        // 📌 تحويل بيانات الورقة إلى مصفوفة JSON حتى نتمكن من تعديلها بسهولة
        const sheetData = XLSX.utils.sheet_to_json(sheet, { header: 1 });
        
        // ➕ إضافة الصف الجديد إلى البيانات الحالية
        sheetData.push(newRow);

        // 🔄 تحويل البيانات مرة أخرى إلى ورقة عمل Excel
        const newSheet = XLSX.utils.aoa_to_sheet(sheetData);
        workbook.Sheets["Operations"] = newSheet;

        // 💾 كتابة الملف بعد التحديث
        const excelBuffer = XLSX.write(workbook, { bookType: "xlsx", type: "buffer" });
        await fs.writeFile(EXCEL_PATH, excelBuffer);

        console.log("✅ Operation logged in Excel!"); // ✅ تأكيد نجاح العملية
    } catch (error) {
        console.error("⚠️ Error logging to Excel:", error); // ⚠️ طباعة الخطأ إن وجد
    }
}

// 🏠 تنفيذ البرنامج
(async () => {
    let model = await loadModel(); // 📥 تحميل النموذج إذا كان متاحًا
    let limits;
    if (!model) {
        model = createModel(); // 🏗️ إنشاء نموذج جديد إذا لم يكن موجودًا
        limits = await trainModel(model); // 🎯 تدريب النموذج على البيانات المتاحة
    }

    // 📩 طلب إدخال البيانات من المستخدم
    const userInputs = await askUserForInputs();
    console.log(`🔢 Inputs received: ${userInputs}`);

    // 🔄 تطبيع المدخلات لتحضيرها للإدخال في النموذج
    const normalizedInputs = normalizeData(userInputs, limits.minInput, limits.maxInput);
    const inputTensor = tf.tensor2d([normalizedInputs], [1, 4]);

    console.log("📊 Predicting house price...");
    const outputTensor = model.predict(inputTensor); // 📈 التنبؤ بالسعر باستخدام النموذج
    const outputArray = outputTensor.arraySync();

    // 🔄 إعادة البيانات إلى نطاقها الطبيعي لحساب السعر الفعلي
    const denormalizedOutput = denormalizeData(outputArray.flat(), limits.minOutput, limits.maxOutput);
    const priceUSD = denormalizedOutput[0]; // 💲 السعر بالدولار
    const priceEGP = priceUSD * EXCHANGE_RATE; // 💰 تحويل السعر إلى الجنيه المصري

    console.log(`🏠 Estimated house price: $${priceUSD.toFixed(2)} (~ EGP ${priceEGP.toFixed(2)})`);

    await generateChart(userInputs, priceUSD, priceEGP); // 📊 إنشاء الرسم البياني
    await logToExcel(userInputs, priceUSD, priceEGP); // 📝 تسجيل البيانات في Excel
})();
