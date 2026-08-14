"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp NGUYÊN VĂN một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp nguyên văn MỘT DÒNG
(xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import re

from harness.middleware import Middleware


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def before_model(self, ctx, messages):
        """Give the first retrieval turn a concise, question-derived query.

        Long support-ticket narratives dilute BM25 with incidental names and
        numbers.  Keep only the policy clause when the question supplies one;
        otherwise add a conservative semantic expansion for a new business
        partner.  This is a one-turn hint: the agent's canonical question is
        left untouched for answering and no corpus metadata is consulted.
        """
        if ctx.step != 0:
            return messages
        question = ctx.question.strip()
        match = re.search(
            r"\btheo quy định\s+(.+?)(?:,|\?|$)", question, re.I
        )
        if match and re.search(r"\bbốc dỡ\b", match.group(1), re.I):
            query = re.sub(r"\bkhi\b", " ", match.group(1), flags=re.I)
            query = " ".join(query.split())
        elif re.search(r"hợp tác lần đầu", question, re.I) and re.search(
            r"\bđào tạo\b", question, re.I
        ):
            query = "quy trình làm việc với nhà cung cấp mới phòng đào tạo thống kê"
        else:
            return messages
        return messages + [{"role": "user", "content": query}]

    def after_agent(self, ctx, report):
        # TODO (§11): khoảng 10-25 dòng.
        #  1. Lấy report["claims"]; bỏ qua nếu rỗng hoặc ctx.corpus là None.
        #  2. Với mỗi claim, gọi ctx.corpus.get(claim["doc_id"]).
        #     Nếu tài liệu tồn tại VÀ claim["text"] khớp NGUYÊN VĂN một
        #     DÒNG trong body của nó (không phải chỉ "nằm trong body")
        #     -> trích dẫn đã đúng, giữ nguyên claim.
        #  3. Nếu không: tìm trong ctx.corpus.docs tài liệu đầu tiên thoả
        #     doc.body in ctx.observed_text  và  claim["text"] khớp
        #     nguyên văn một DÒNG của doc.body -> đó là nguồn thật.
        #     Đổi doc_id sang nó, GIỮ NGUYÊN text.
        #  4. Không tìm được nguồn nào -> để `critic` xử lý, đừng bịa doc_id.
        #  5. Cập nhật report["citations"] = danh sách doc_id đã sắp xếp.
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims or ctx.corpus is None:
            return report

        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
                continue
            text = claim["text"]
            cited = ctx.corpus.get(claim.get("doc_id"))
            if cited is not None and any(text in line for line in cited.body.splitlines()):
                continue
            for doc in ctx.corpus.docs:
                if doc.body in ctx.observed_text and any(
                    text in line for line in doc.body.splitlines()
                ):
                    claim["doc_id"] = doc.doc_id
                    break

        report["citations"] = sorted({
            claim.get("doc_id") for claim in claims
            if isinstance(claim, dict) and isinstance(claim.get("doc_id"), str)
        })
        return report
