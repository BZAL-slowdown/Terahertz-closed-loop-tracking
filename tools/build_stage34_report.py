"""生成阶段三、四复现报告（Word）。

图来自 outputs/stage3_4/figures，数值来自同目录的 JSON/CSV。
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "outputs" / "stage3_4" / "figures"
OUT = ROOT / "docs" / "220GHz_THz_stage3_4_report.docx"

BODY_CN = "宋体"
HEAD_CN = "黑体"
LATIN = "Times New Roman"
BODY_PT = 10.5
TEXT_WIDTH_CM = 14.2
LINE_SPACING = 1.34


def set_run_font(run, east=BODY_CN, latin=LATIN, size=BODY_PT, bold=False, color=None):
    run.font.name = latin
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    fonts.set(qn("w:ascii"), latin)
    fonts.set(qn("w:hAnsi"), latin)
    fonts.set(qn("w:eastAsia"), east)


def style_font(style, east, size, bold=False):
    style.font.name = LATIN
    style.font.size = Pt(size)
    style.font.bold = bold
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    fonts.set(qn("w:ascii"), LATIN)
    fonts.set(qn("w:hAnsi"), LATIN)
    fonts.set(qn("w:eastAsia"), east)


def add_field(paragraph, instruction, size=9, color=RGBColor(0x66, 0x66, 0x66)):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)
    set_run_font(run, size=size, color=color)


def body(document, text, indent=True, space_after=4):
    paragraph = document.add_paragraph()
    fmt = paragraph.paragraph_format
    fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt.line_spacing = LINE_SPACING
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(space_after)
    if indent:
        fmt.first_line_indent = Pt(BODY_PT * 2)
    for index, chunk in enumerate(text.split("**")):
        if chunk:
            set_run_font(paragraph.add_run(chunk), bold=index % 2 == 1)


def heading(document, text, level):
    paragraph = document.add_paragraph(style=f"Heading {level}")
    run = paragraph.add_run(text)
    if level == 1:
        set_run_font(run, east=HEAD_CN, size=14, bold=True, color=RGBColor(0x1F, 0x1F, 0x1F))
    else:
        set_run_font(run, east=HEAD_CN, size=11.5, bold=True, color=RGBColor(0x2B, 0x2B, 0x2B))
    paragraph.paragraph_format.space_before = Pt(12 if level == 1 else 9)
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.keep_with_next = True


def figure(document, filename, caption):
    path = FIG / filename
    if not path.exists():
        raise FileNotFoundError(path)
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Cm(TEXT_WIDTH_CM))

    cap = document.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)
    set_run_font(cap.add_run(caption), size=9, color=RGBColor(0x40, 0x40, 0x40))


def _shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def table(document, caption, headers, rows, width_cm=13.8):
    cap = document.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(3)
    cap.paragraph_format.keep_with_next = True
    set_run_font(cap.add_run(caption), size=9, color=RGBColor(0x40, 0x40, 0x40))

    tbl = document.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, text in enumerate(headers):
        cell = tbl.rows[0].cells[index]
        _shade(cell, "E8F0F1")
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(1)
        paragraph.paragraph_format.space_after = Pt(1)
        set_run_font(paragraph.add_run(text), east=HEAD_CN, size=9, bold=True)

    for row in rows:
        cells = tbl.add_row().cells
        for index, text in enumerate(row):
            paragraph = cells[index].paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_before = Pt(1)
            paragraph.paragraph_format.space_after = Pt(1)
            set_run_font(paragraph.add_run(text), size=9)

    for column in tbl.columns:
        for cell in column.cells:
            cell.width = Cm(width_cm / len(headers))


def build():
    document = Document()
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.4)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.8)

    normal = document.styles["Normal"]
    style_font(normal, BODY_CN, BODY_PT)
    normal.paragraph_format.line_spacing = LINE_SPACING
    for level in (1, 2):
        style_font(document.styles[f"Heading {level}"], HEAD_CN, 14 if level == 1 else 11.5, bold=True)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_run_font(header.add_run("220 GHz 太赫兹跟瞄 · 阶段三四复现"),
                 size=8.5, color=RGBColor(0x88, 0x88, 0x88))
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(footer, "PAGE")

    title = document.add_paragraph()
    title.paragraph_format.space_after = Pt(2)
    set_run_font(title.add_run("220 GHz 太赫兹跟瞄：阶段三、四复现"),
                 east=HEAD_CN, size=20, bold=True)

    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(4)
    set_run_font(subtitle.add_run("双向时分交替、被动标量退化与正交移动跟踪"),
                 east=HEAD_CN, size=12, color=RGBColor(0x44, 0x44, 0x44))

    meta = document.add_paragraph()
    meta.paragraph_format.space_after = Pt(12)
    set_run_font(meta.add_run("2026 年 10 月 30 日　　代码：code/stage3_4_thz.py"),
                 size=9, color=RGBColor(0x77, 0x77, 0x77))
    rule = document.add_paragraph()
    ppr = rule._p.get_or_add_pPr()
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "1F6F7A")
    border.append(bottom)
    ppr.append(border)
    rule.paragraph_format.space_after = Pt(6)

    heading(document, "一、这一轮做了什么", 1)
    body(document,
         "阶段一二解决的是“单端、静态、能不能收敛”。这一轮把双向时分（阶段三）和正交移动（阶段四）"
         "放进同一套循环里，用速度开关分两次验证：速度因子 0 时跑 160 拍，两端各 Active 80 拍；"
         "速度因子 0.2 时跑 200 拍，Rover 沿 y 水平走、Elevator 沿 z 垂直爬。")
    body(document,
         "算法骨架沿用我手写的 MATLAB 版本：8 个扫描点一圈、两端互斥、被动端半径与增益门控清零、"
         "被动期只做标量更新。改的只有三处：信道换成实测方向图加 P.676，滤波器从 3 状态扩到 6 状态，"
         "角度量纲按波束宽度重标。")
    body(document,
         "**量纲必须交代清楚。** 作者的高斯波束半功率宽度 35.3°，实测太赫兹只有 0.768°，比例 0.0217。"
         "所以初始偏差 8°/−6° 换成 0.174°/−0.130°，扫描半径 2°–5° 换成 0.043°–0.109°，"
         "再按 0.11 个半功率宽度封顶到 0.084°。直接把 2°–5° 搬过来不行：0.77° 的波束上，"
         "2° 已经是 2.6 倍半功率宽度，实测方向图在那个位置是 20–30 dB 的边瓣，斜率会翻号。")
    table(document, "表 1　阶段三四参数",
          ["项目", "取值", "说明"],
          [["每块扫描点 ns", "8", "360°/45°"],
           ["采样周期 dt", "0.5 s", "与论文 Fig.7 一致"],
           ["静态步数", "160", "两端各 Active 80 拍"],
           ["动态步数", "200", "速度因子 0.2"],
           ["扫描半径", "0.043–0.084°", "LED 2°–5° 按 HPBW 重标"],
           ["初始偏差", "0.174 / −0.130°", "LED 8°/−6° 重标"],
           ["回中增益", "1.0", "块末一次回中"]])

    heading(document, "二、双向时分与被动标量防线", 1)
    body(document,
         "时分调度是硬约束：任一时刻有且只有一端在扫。被动端的扫描半径直接置零、PI 增益置零，"
         "只保留一个 0 偏置的功率样本，用来更新基线 q。这一步看起来只是“少更新几个状态”，"
         "实际上是两件事：不做 6×6 求逆，也不让角度和 q 的交叉项在无观测时被反复放大。")
    body(document,
         "具体做法是式(27)-(31)的六状态推广。被动期角度四个状态（两轴误差、两轴角速度）与它们的"
         "4×4 协方差块原样带过去，q 只做标量更新，增益 K1 = P11/(P11+R_eff)，q 行与 q 列按 1−K1 "
         "同步缩放。这样 P 仍是半正定的，下一拍预测不会崩，而角度块一个元素都没动。"
         "160 拍里实测的最大偏差是 0，断言门限 1e-12，一次都没触发。")
    body(document,
         "有两个坑是调出来的，值得写下来。第一个：被动期不能拿 q_dot 外推 q。q_dot 在被动期是冻的，"
         "只要它偏 0.3 dB/s，8 拍就能把基线推走 1.2 dB，转主动后滤波器会把这 1.2 dB 当成角度损耗，"
         "V 轴直接反号。作者原文 x1_forecast = x1，不做动力学外推，这里照做。")
    body(document,
         "第二个：被动样本里混着“角度未知”造成的损耗，不能全算到 q 头上。一开始按局部梯度估这个"
         "不确定度，结果峰顶梯度为零，等于没估，8 次更新把 P11 压死在一个偏了 1.6 dB 的 q 上，"
         "R2 一上台就发散。改成用 ±1σ 的指向不确定度去量损耗展布（峰顶处梯度为零，但偏 1σ 可能"
         "就是好几 dB），K1 随之变小，q 在被动期基本不动，转主动后交给 8 点批量更新去分。")
    body(document,
         "半径这块，式(25)(26)是窗口内 P_ang 最大特征值求和再限幅。作者的 P 被 Q_scaling=1e5 放大过，"
         "特征值在 1e4 deg² 量级，乘 K_delta=1/(7000·ns) 正好落在 2°–5°；我们的滤波器用真实 deg²，"
         "特征值只有 1e-4 量级，直接乘会恒等于零。所以保留“窗口求和 + 限幅”的结构，把尺度换成"
         "角度标准差 sqrt(λ_max)，系数取 2.2。图 1 里两端的半径此起彼伏，R2 从 0.084° 一路收到 0.047°，"
         "就是它在收敛的直接证据。")
    figure(document, "fig1_radius_pulse.png", "图 1　双向交替下的扫描半径脉冲（静态，160 拍）")

    heading(document, "三、正交移动的单次闭环", 1)
    body(document,
         "速度因子 0.2 对应每步位移 0.707 mm，100 秒里距离从 2.3000 m 走到 2.3087 m，"
         "和作者 Fig.9 的构型一致。这个位移换算成视距角速度只有 0.02°/s，一圈扫描 4 秒转 0.08°，"
         "约为 0.1 个半功率宽度——听起来不大，但被动端的 4 秒盲区里没人纠正，误差就靠这个速度一点点攒。")
    body(document,
         "结果是 Rover 末端 0.122°、Elevator 0.309°，末 10 拍 RMS 分别为 0.174° 和 0.348°。"
         "Elevator 差一倍多，不是调度写错了，是它的被动窗口刚好压在视距俯仰变化最快的那一段；"
         "等它上台，误差已经攒到 0.3° 以上，8 点一圈只能拉回来一部分。图 3(b) 里那条锯齿就是这个节奏："
         "块内往上爬，块末往下掉。")
    figure(document, "fig3_dynamic_tracking.png", "图 3　正交移动下的距离变化与两端指向误差")

    heading(document, "四、与作者 LED 数据的对标", 1)
    body(document,
         "图 2 和图 4 重绘的是作者归档的 MAT 数据，信道还是 LED 的高斯模型，不是这一轮的太赫兹结果。"
         "放在这里是为了给两条趋势找参照，不能拿绝对数直接比。")
    body(document,
         "距离一组：EKF 在 0.92 m 是 0.899°，到 5.06 m 抬头到 5.166°；传统 ES 在 0.92 m 直接崩到"
         "28.250°，之后被限幅在 3° 附近。近距离的崩溃是“高斜率悬崖”的典型表现——波束太窄，"
         "固定步长的梯度法一步就跨过主瓣，越调越远。")
    body(document,
         "速度一组：0 到 1.3 cm/s，EKF 从 0.998° 涨到 19.511°，1 cm/s 以后基本是线性抬头；"
         "ES 那条线一直贴着 3°，看着“稳”，其实是根本没跟上，误差被限幅器钉住了。"
         "EKF 的抬头来自时标错配：扫描一圈 4 秒，这段时间目标已经走了 5 mm，"
         "准静态假设下的预测和线性化点都对不上，误差只能越滚越大。")
    figure(document, "fig2_distance_reference.png", "图 2　作者 MAT 数据：距离—误差（LED 基线）")
    figure(document, "fig4_speed_reference.png", "图 4　作者 MAT 数据：速度—误差（LED 基线）")
    table(document, "表 2　本轮太赫兹结果与作者 LED 基线",
          ["场景", "本轮（THz）", "作者（LED）"],
          [["静态双向，R1 末段 RMS", "0.0079°", "—"],
           ["静态双向，R2 末段 RMS", "0.0188°", "—"],
           ["正交移动，Rover 末端", "0.122°", "0.816°"],
           ["正交移动，Elevator 末端", "0.309°", "1.198°"],
           ["距离 0.92→5.06 m", "未扫", "0.899→5.166°"],
           ["速度 0→1.3 cm/s", "未扫", "0.998→19.511°"],
           ["被动冻结最大偏差", "0（<1e-12）", "—"]])

    heading(document, "五、几个判断", 1)
    body(document,
         "**半径不能跨波束搬。** 2°–5° 是给 35° 波束用的，换到 0.77° 的波束上，2° 已经出了主瓣。"
         "自适应律本身没问题，问题是上下限得跟着波束走；这一轮的做法是按 HPBW 重标，"
         "保留式(25)(26)的结构，把无量纲部分留给 λ_max。")
    body(document,
         "**被动端守得住，是因为什么都没算。** 没有 6×6 求逆、没有交叉项参与更新，"
         "角度就按物理直觉停在原地；代价是这 4 秒里目标怎么动，它完全不知道。"
         "所以被动期的长度，本质上就是系统的“盲区时长”，速度一高，误差就按这个时长线性攒。")
    body(document,
         "**速度的边界是可以估的。** 盲区 4 秒、允许误差 0.2 个 HPBW（约 0.15°），"
         "对应视距角速度上限大约 0.04°/s。速度因子 0.2 时是 0.02°/s，还有一倍余量；"
         "再往上加，锯齿的峰值就会顶出主瓣，和作者 Fig.10 的抬头是同一个机理。")
    body(document,
         "**这一轮的太赫兹绝对误差比 LED 小一个量级，不代表算法更好。** 波束窄了 46 倍，"
         "同样的角度误差在功率上被放大几十倍，闭环自然更“敏感”；但捕获域也同比缩小，"
         "初始偏差稍大一点就进不去。两个数不能直接比大小，只能比趋势。")

    heading(document, "六、下一步", 1)
    body(document,
         "最直接的补丁是 IMU 前馈。被动端的 4 秒盲区里，角度靠“冻结”硬扛；如果平台上有陀螺，"
         "这段时间的姿态变化可以用短时惯性积分补出来，作为前馈加到云台指令上，"
         "被动期就不再是完全的盲区。惯导的高频噪声不进滤波器，只走前馈通路，"
         "和 8 点一圈的低频功率观测形成互补——低频管绝对指向，高频管短时保持。")
    body(document,
         "第二个补丁是把圆周扫描拆成粗精两级：粗扫用大半径快速捕获，精扫用小半径压噪声，"
         "半径的切换阈值直接用 P_ang 的最大特征值。这一轮的自适应半径只做了“连续收”，"
         "还没做“该放的时候放”，捕获阶段仍然靠初始半径硬顶。")
    body(document,
         "工程节奏上，先把 Passive 分支并回主仿真器（现在是独立脚本），再挂 IMU 接口做半实物，"
         "最后上转台。顺序反了会白跑：转台上的振动谱一旦进来，先暴露的肯定是盲区，而不是滤波器。")

    heading(document, "七、结论", 1)
    body(document,
         "阶段三、四在 220 GHz 实测信道上跑通了：双向时分的被动期做到逐位冻结（实测偏差 0，"
         "门限 1e-12），自适应半径随协方差收缩，正交移动下两端末端误差 0.122° 和 0.309°，"
         "都落在主瓣以内。作者的 LED 基线（0.816°/1.198°、0.899→5.166°、0.998→19.511°）"
         "作为趋势参照保留，不作为太赫兹的定量结论。")
    body(document,
         "代码在 code/stage3_4_thz.py，静态与动态各一条命令；图在 outputs/stage3_4/figures，"
         "原始逐拍数据是同目录的 CSV。", indent=True, space_after=0)

    document.save(OUT)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    build()
