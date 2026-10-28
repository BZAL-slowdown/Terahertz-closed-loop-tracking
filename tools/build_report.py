"""生成阶段一二复现的 Word 汇报（结果版）。

四张结果图 + 两张汇总表；数值与 outputs_stage1_2_figures 下的实际运行一致。
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
FIG = ROOT / "outputs" / "figures"
OUT = ROOT / "docs" / "220GHz_THz_stage1_2_report.docx"

BODY_CN = "宋体"
HEAD_CN = "黑体"
LATIN = "Times New Roman"
BODY_PT = 10.5
TEXT_WIDTH_CM = 14.2
LINE_SPACING = 1.34


def set_run_font(run, east=BODY_CN, latin=LATIN, size=BODY_PT, bold=False,
                 color: RGBColor | None = None, italic=False) -> None:
    run.font.name = latin
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
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


def style_font(style, east, size, bold=False, color=None) -> None:
    style.font.name = LATIN
    style.font.size = Pt(size)
    style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    fonts.set(qn("w:ascii"), LATIN)
    fonts.set(qn("w:hAnsi"), LATIN)
    fonts.set(qn("w:eastAsia"), east)


def add_field(paragraph, instruction: str, size=9, color=RGBColor(0x66, 0x66, 0x66)) -> None:
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


def body(document: Document, text: str, indent: bool = True, space_after: float = 4) -> None:
    paragraph = document.add_paragraph()
    fmt = paragraph.paragraph_format
    fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt.line_spacing = LINE_SPACING
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(space_after)
    if indent:
        fmt.first_line_indent = Pt(BODY_PT * 2)
    for index, chunk in enumerate(text.split("**")):
        if not chunk:
            continue
        run = paragraph.add_run(chunk)
        set_run_font(run, bold=index % 2 == 1)


def heading(document: Document, text: str, level: int) -> None:
    paragraph = document.add_paragraph(style=f"Heading {level}")
    run = paragraph.add_run(text)
    if level == 1:
        set_run_font(run, east=HEAD_CN, size=14, bold=True, color=RGBColor(0x1F, 0x1F, 0x1F))
    else:
        set_run_font(run, east=HEAD_CN, size=11.5, bold=True, color=RGBColor(0x2B, 0x2B, 0x2B))
    paragraph.paragraph_format.space_before = Pt(12 if level == 1 else 9)
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.keep_with_next = True


def figure(document: Document, filename: str, caption: str) -> None:
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
    cap.paragraph_format.space_before = Pt(0)
    cap.paragraph_format.space_after = Pt(8)
    run = cap.add_run(caption)
    set_run_font(run, size=9, color=RGBColor(0x40, 0x40, 0x40))


def _shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def table(document: Document, caption: str, headers: list[str], rows: list[list[str]],
          width_cm: float = 13.6) -> None:
    cap = document.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(3)
    cap.paragraph_format.keep_with_next = True
    set_run_font(cap.add_run(caption), size=9, color=RGBColor(0x40, 0x40, 0x40))

    tbl = document.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True
    header_cells = tbl.rows[0].cells
    for index, text in enumerate(headers):
        _shade(header_cells[index], "E8F0F1")
        paragraph = header_cells[index].paragraphs[0]
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

    tail = document.add_paragraph()
    tail.paragraph_format.space_before = Pt(0)
    tail.paragraph_format.space_after = Pt(2)


def build() -> None:
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
        style_font(document.styles[f"Heading {level}"], HEAD_CN, 14 if level == 1 else 11.5,
                   bold=True)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = header.add_run("220 GHz 太赫兹跟瞄 · 阶段一二复现")
    set_run_font(run, size=8.5, color=RGBColor(0x88, 0x88, 0x88))
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(footer, "PAGE")

    title = document.add_paragraph()
    title.paragraph_format.space_after = Pt(2)
    set_run_font(title.add_run("220 GHz 太赫兹跟瞄阶段一二复现"),
                 east=HEAD_CN, size=20, bold=True)

    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(4)
    set_run_font(subtitle.add_run("实测方向图信道、大气预算与静态闭环跟踪结果"),
                 east=HEAD_CN, size=12, color=RGBColor(0x44, 0x44, 0x44))

    meta = document.add_paragraph()
    meta.paragraph_format.space_after = Pt(12)
    set_run_font(meta.add_run("2026 年 10 月 28 日　　数据与图：outputs/figures"),
                 size=9, color=RGBColor(0x77, 0x77, 0x77))
    rule = document.add_paragraph()
    rule.paragraph_format.space_after = Pt(6)
    ppr = rule._p.get_or_add_pPr()
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "1F6F7A")
    border.append(bottom)
    ppr.append(border)

    heading(document, "一、复现范围", 1)
    body(document,
         "这次复现阶段一和阶段二。阶段一把论文里解析的高斯光强面换成 220 GHz 实测方向图，"
         "并按 ITU-R P.676 计入氧气与水汽吸收；阶段二在这套信道上做单端、静态视距的三点扫描闭环。"
         "参数固定：dt=0.5 s，200 拍共 100 s，测量噪声 0.1 dB，随机种子 20261028，"
         "距离取 0.92 m 和 15 m。双向交替、Passive 冻结和移动平台属于阶段三四，这次没有开展。")

    heading(document, "二、实现要点", 1)
    body(document,
         "信道取工作簿里 H、V 两条切面，各 601 点、0.02° 一个点。按各自峰值对齐到零偏角后算相对损耗，"
         "用 PCHIP 插值；没有用三次样条，因为样条在峰值附近会过冲，插出高于实测峰值的增益，"
         "零点附近的斜率随之改变。二维方向图按 L(α,ε)=L_H(α)+L_V(ε) 可分离近似，"
         "交叉耦合不可见是这个近似的代价。超出测量域时损耗夹紧、导数置零，避免外推值被当成测量。")
    body(document,
         "大气按 P.676 逐线计算，氧气 0.015444、水汽 2.471371 dB/km，再乘距离。"
         "滤波器是六状态 q、q_dot、两轴误差、两轴角速度；预测用常数功率变化率与常数角速率，"
         "量测为异步三点 z_i = q − q_dot·τ_i + g(e − ω·τ_i − s_i)。1×6 雅可比由 PCHIP 导数解析给出，"
         "中心差分核验的最大偏差 2.3e-9；协方差用 Joseph 形式，迭代重线性化 8 次。"
         "闭环每拍三点扫描、更新、回中，半径固定为 0.092°（0.12 倍平均 HPBW）。")

    heading(document, "三、复现结果", 1)
    heading(document, "3.1 信道与链路预算", 2)
    body(document,
         "图 1 是复现出的二维波束与三个扫描点的位置。半功率宽度 H 面 0.703°、V 面 0.833°，"
         "峰值增益 44.96 和 44.69 dBi，三个扫描点都在 ±0.10° 的主瓣肩上。"
         "图 2 是链路预算：0.92 m 与 15 m 的接收功率分别为 31.07 和 6.79 dBm，SNR 108.0 和 83.7 dB，"
         "相差 24.28 dB；15 m 处气体吸收只有 0.037 dB，差距几乎全部来自自由空间损耗。"
         "表 1 汇总了这组数值。")
    figure(document, "fig01_beam_map_2d.png", "图 1　实测方向图复现与三点扫描位置")
    figure(document, "fig04_link_budget_distance.png", "图 2　链路预算随距离的变化")
    table(document, "表 1　阶段一信道与链路预算",
          ["项目", "0.92 m", "15 m"],
          [["峰值增益 H / V（dBi）", "44.96 / 44.69", "44.96 / 44.69"],
           ["3 dB 波束宽度 H / V（°）", "0.703 / 0.833", "0.703 / 0.833"],
           ["自由空间损耗（dB）", "78.57", "102.82"],
           ["氧气 + 水汽吸收（dB）", "0.0023", "0.0373"],
           ["接收功率（dBm）", "31.07", "6.79"],
           ["信噪比（dB）", "108.0", "83.7"]])

    heading(document, "3.2 静态闭环跟踪", 2)
    body(document,
         "图 3 是 0.92 m 下的闭环结果。初始指向误差 H 0.084°、V −0.083°，范数 0.118°；"
         "第一拍回中后误差进入 ±0.03°，10 s 滑动 RMS 从 0.119° 降到 0.0071°，约 17 倍。"
         "末 20 拍 RMS 为 0.0070°，终值 0.0120°，200 拍测量全部被滤波器接受，协方差全程正定。"
         "表 2 给出闭环指标。15 m 的结果与 0.92 m 逐点一致，差别只在功率基线 q 上，"
         "原因在第四节说明。")
    figure(document, "fig09_static_result.png", "图 3　静态闭环跟踪结果（0.92 m，100 s）")
    table(document, "表 2　阶段二静态闭环指标",
          ["指标", "数值"],
          [["初始误差范数（回中前）", "0.1185°"],
           ["末 20 拍 RMS（回中后）", "0.0070°"],
           ["滑动 RMS 末值（回中前）", "0.0071°"],
           ["单拍终值（回中前）", "0.0120°"],
           ["单拍终值（回中后）", "0.0047°"],
           ["IEKF 测量接受", "200 / 200"],
           ["NEES 均值（后 60 拍）", "0.42"]])

    heading(document, "3.3 反馈律选择", 2)
    body(document,
         "图 4 比较了三种反馈律，滤波器、测量与随机种子完全相同。末 20 拍 RMS："
         "原 PI 0.0367°、单位反馈 0.0070°、固定步长梯度上升 13.8°。"
         "PI 的指令曲线在死区内被冻成恒定的位置增量，误差呈锯齿状，0.054° 那个数其实是极限环上的采样点，"
         "不是稳态。主程序的云台有 0.1° 量化，积分用于跨越死区；阶段一二的简化云台没有量化，"
         "因此这里取单位反馈。")
    figure(document, "fig06_control_laws.png", "图 4　三种反馈律的闭环结果对比")

    heading(document, "四、讨论", 1)
    body(document,
         "q 的作用不是吸收误差。三点扫描的角度信息来自样本之间的功率差，q 承担其中的公共项；"
         "没有这个状态，幅值变化会被误判为角度偏差，角度估计随之漂移。")
    body(document,
         "扫描半径应当按轴选。V 面在 0.10–0.17° 有一段斜率只有 0.76 dB/deg 的平肩，"
         "而 H 面在同样半径处有 5.0 dB/deg；按 HPBW 比例统一取半径，V 轴拿到的角度信息明显偏少。"
         "按 3 dB 主瓣内最陡点算，H 在 0.249°、V 在 0.428°。")
    body(document,
         "这组参数下测角精度与热噪声无关。0.1 dB 的测量噪声比辐射计噪声大几个量级，"
         "要到 2.1 km 两者才相当，所以两个距离的角误差逐点相同，差别全在 q 上。"
         "若要讨论距离对精度的影响，需把 σ 换成由 SNR 推导的量。")
    body(document,
         "闭环收敛后的误差底由估计噪声决定，与初始条件无关：初始偏差从 0.006° 加到 0.18°，"
         "末 20 拍 RMS 都停在 0.0070° 附近。这意味着继续压低误差要靠更好的测量，"
         "而不是更快的回中；单位反馈把估计值直接当指令，估计噪声也会进入指令。")

    heading(document, "五、结论与遗留", 1)
    body(document,
         "阶段一二在 220 GHz 实测信道上复现完成：链路预算、六状态 IEKF 和静态闭环都给出了可复现的数值，"
         "闭环末段误差 0.0070°，相当于 0.009 个半功率宽度。")
    body(document,
         "遗留两项。其一，主程序缺少论文 III-D 的 Passive 状态机，step() 中两端滤波器每拍都 predict，"
         "非活跃端没有量测却持续累积过程噪声，协方差只涨不消，因此阶段三四的双向交替、冻结断言与"
         "移动跟踪都还没有做。其二，0.92 m 的旧摘要缺少 peak_received_power_dbm 等三个字段，"
         "已按当前代码重跑至 outputs/stage1_2_0.92m。")
    body(document,
         "代码入口：太赫兹/stage1_2_figures.py；--only 指定单张图，--distance-m 更换距离，"
         "图表数值与 figure_summary_0.92m.json 一致。", indent=True, space_after=0)

    document.save(OUT)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    build()
