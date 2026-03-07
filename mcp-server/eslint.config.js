import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import prettier from "eslint-config-prettier";

export default tseslint.config(
    eslint.configs.recommended,
    ...tseslint.configs.strictTypeChecked,
    ...tseslint.configs.stylisticTypeChecked,
    prettier,
    {
        languageOptions: {
            parserOptions: {
                projectService: true,
                tsconfigRootDir: import.meta.dirname,
            },
        },
        rules: {
            "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
            "@typescript-eslint/no-explicit-any": "error",
            "@typescript-eslint/no-non-null-assertion": "error",
            "@typescript-eslint/explicit-function-return-type": "error",
            "@typescript-eslint/explicit-module-boundary-types": "error",
            "@typescript-eslint/no-floating-promises": "error",
            "@typescript-eslint/no-misused-promises": "error",
            "@typescript-eslint/strict-boolean-expressions": "error",
            "@typescript-eslint/switch-exhaustiveness-check": "error",
            "@typescript-eslint/no-unnecessary-condition": "error",
            "@typescript-eslint/prefer-nullish-coalescing": "error",
            "@typescript-eslint/prefer-optional-chain": "error",
            "@typescript-eslint/consistent-type-imports": [
                "error",
                { prefer: "type-imports" },
            ],
            "@typescript-eslint/consistent-type-definitions": ["error", "interface"],
            "@typescript-eslint/naming-convention": [
                "error",
                { selector: "default", format: ["camelCase"] },
                { selector: "variable", format: ["camelCase", "UPPER_CASE"] },
                {
                    selector: "parameter",
                    format: ["camelCase"],
                    leadingUnderscore: "allow",
                },
                { selector: "typeLike", format: ["PascalCase"] },
                { selector: "enumMember", format: ["PascalCase"] },
                { selector: "import", format: null },
            ],
            "no-console": ["error", { allow: ["error"] }],
            eqeqeq: ["error", "always"],
            "no-var": "error",
            "prefer-const": "error",
            "no-throw-literal": "error",
            curly: ["error", "all"],
        },
    },
    {
        ignores: ["build/", "node_modules/", "eslint.config.js"],
    },
);
