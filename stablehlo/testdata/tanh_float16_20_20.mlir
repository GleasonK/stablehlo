// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xf16>
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = stablehlo.tanh %0 : tensor<20x20xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<20x20xf16>, tensor<20x20xf16>) -> ()
    return %2 : tensor<20x20xf16>
  }
  func.func private @inputs() -> (tensor<20x20xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x583FB4C099C304BC80C1A33EAE2FFCC5953EA0B5373603445B43B9415041C03466B7553EE6C10FBE81416EC3AFC41BB5AD4308B5513C1246883915C0E0B836B820C340411444FBC04EC0B9C149A47EC0C4C04D3FF7C414C3F1C14DC691BB74C240BC6B42C1C266C05433B23AD33EF44467C6854055C620C22245F433683DD5B4DBBFC93C9642303B63C1B6C33534963C663D3CC236BE6FBDCDBF9544773E12C210415D41EABEC6C0CDC1B8B950C4743C813E2D399238063ABDBAD1BE043FAB388344E0C4EFC7FE44DF41B6BCB3420235E0392846ECBA19BA473C2AC5033CB33AA23D31BE03C6D73C844456340F307842FB42E246A84178BB66C43945AD4029B7404545B30FC265B96B4237372A4055C2913EED35CCBC953C023F1344103C57B421C597422C458D39FAC007BDC8448ABA39C2753A5D40BF42173D81C16B418D44C23672B31D43D13D7AC1E84005C60342F8BC75B4FB4180BEA4433EC8973E9DBD2BC0F6AC9FB5272BB53447C4D03E15C4FCC2D33DEAB588BDE8C044C073ABB6B61B3DD5C12EBFFEC21CB347C03CC0DAB974BEE3BC094457C4043DE0AA11BE07C537C20BBF9A3DB341BE33A9C2FC3F9CAA0BBE682E4B4161436F42FABBF3AEC23CB544DD3D7EBB583DC2BED4320444F1C10545DAC439C0304635B8B2C321BD784177BD803DFC45FBC3CAC05EB5B945B7C74144A0B8D538FCB72C4004C31439B4BD40B207C4D4B8FF42E1C444C437383E3D13BC9D41A244363C5EC0ADC082B9F3BDB53D9DC8D0417AB901C47FBE6F4527BE6BC405412EBB2C40043EA8BA83A70BB0273FACBD48BC2AB576BD103914C57C417BB3E4C6154801406DBF45C167C41D33B03C1645C74493BBA64413447A3BA64002BE2D3570B0EDBB9044283ECBC3B744EAC43E414FBBC0C489C07A22B13F4B33F4BA0B41053FCDC2AD44DE421C39BA40ED41F0C10F352FC629BF20B3A4456D3DF2C39E42013F0D41AAB40A435C43BBAB83409ABD3DC68EBD96BC80C35F307BBF31AF5EBC4BC35AB92BBB9743BCC59FC44AC37142F6B51E3C63C23D3C25423CC372B48641D0C01E41AFB944248D4205C5C738B139AEAE7FC498C4B938C8BA3242FCC545B9DD47A740E9C1AD3CB1C223BD"> : tensor<20x20xf16>
    return %cst : tensor<20x20xf16>
  }
  func.func private @expected() -> (tensor<20x20xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x9A3BDBBBFEBB1BBAEFBB713BA52F00BC6D3B67B5EC35FF3BFD3BF33BEC3B9D34EAB65A3BF5BB43BBEF3BFEBB00BCF0B4FE3BDFB4583A003CCB38BCBB59B8B9B7FDBBEB3BFF3BE4BBC9BBF3BB49A4D3BBDDBB983B00BCFDBBF5BB00BCE7B9FABB4BBAF93BFBBBCEBB343379397D3B003C00BCD43B00BCF7BB003CCB33FF3AB1B4B1BBA93AFA3BB939EDBBFEBB1D34883AFE3AF8BB50BB02BBAFBB003C653BF7BBE63BED3B83BBDEBBF4BBE9B8FFBB713A673B8F38213818397FB97DBB893B3438003C00BC00BC003CF43B9DBAFB3BDA340239003C97B924B9503A00BC1A3A7A39193B4FBB00BCB13A003C3C340930FA3BFC3B003CF23BDCB9FFBB003CDA3BB8B6003C26B3F6BBB4B8F93BC336C13BF9BB6C3BAB35ABBA883A883BFF3B253A3DB400BCFA3B003CCE38E4BBCDBA003C64B9F8BB5839CC3BFB3BD63AEFBBEE3B003C623650B3FD3B2C3BEFBBE23B00BCF63BC5BA58B4F63B67BBFE3B00BC6E3B17BBC2BBF3AC67B5252B9334FFBB7D3BFFBBFCBB2D3BA9B50DBBE2BBC7BB71AB58B6D83AF4BB92BBFCBBFFB2C8BBC6BBFEB864BBB9BAFF3BFFBBCC3ADEAA44BB00BCF8BB8ABB153BF23B9833FBBBB63B9AAA42BB632EEC3BFD3BF93B15BAECAEA53A003C313BDEB9F73A79BBBA32FF3BF5BB003C00BCC5BB003CB7B7FEBBDBBAEF3B06BB0A3B003CFFBBDEBB2DB5003C00BCFF3B2CB8523862B7C23BFCBB7E3820BB2CB2FFBB51B8FC3B00BCFFBBBA37EA3A27BAF13B003C433ACDBBDABBC7B839BB213B00BCF43BC2B8FFBB67BB003C4BBBFFBBE53BB8B9C23B3F3B74B982A706B0913B1DBB51BAFEB405BB7B3800BCEF3B59B300BC003CB73B9EBBEBBBFFBB0033993A003C003CE8B9003CFF3BDD39D93B3EBB013569B010BA003C4C3BFEBB003C00BCEA3BC8B900BCD5BB7A22AA3B2B339CB9E63B893BFBBB003CFC3B8338DC3BF53BF5BBE63400BC91BB02B3003C013BFFBBFB3B883BE63B89B4FC3BFD3BB9ABD43B15BB00BC10BB88BAFEBB5830A1BB29AF61BAFDBBADB8B7B9FE3B00BC00BCFDBBF93BB3B5303AF9BB493AF73BFDBB56B4F03BDFBBE83BE3B84424FA3B00BC4838E438A8AEFFBB00BC3E3885B9F83B00BC9FB8003CD93BF5BB973AFBBBDCBA"> : tensor<20x20xf16>
    return %cst : tensor<20x20xf16>
  }
}
